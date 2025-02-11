from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
import click
import frontmatter
import yaml
import re
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
import json

@dataclass
class MarkdownMetadata:
    summary: str
    context: str  # personal, work, research
    next_actions: List[str]
    key_concepts: List[str]
    last_processed: datetime
    word_count: int
    links: List[str]  # Wiki-style links
    status: str  # active, archived, draft

class MarkdownProcessor:
    def __init__(self, model_name: str = "deepseek-r1:8b", verbose: bool = False, skip_recent_days: Optional[int] = None):
        self.llm = OllamaLLM(model=model_name, temperature=0)
        self.verbose = verbose
        self.skip_recent_days = skip_recent_days
        
    def should_process_file(self, post: frontmatter.Post) -> bool:
        """Check if file should be processed based on last_processed date."""
        if not self.skip_recent_days:
            return True
            
        last_processed_str = post.metadata.get('last_processed')
        if not last_processed_str:
            return True
            
        try:
            last_processed = datetime.fromisoformat(last_processed_str)
            cutoff_date = datetime.now() - timedelta(days=self.skip_recent_days)
            should_process = last_processed < cutoff_date
            
            if self.verbose and not should_process:
                click.echo(f"Skipping recently processed file (last processed: {last_processed_str})")
                
            return should_process
        except ValueError:
            # If date parsing fails, process the file
            return True
    
    def extract_metadata(self, content: str) -> MarkdownMetadata:
        """Extract metadata from markdown content using LLM."""
        prompt = PromptTemplate.from_template("""
        Analyze this markdown content and extract metadata in JSON format with these fields:
        - summary: A brief 2-3 sentence summary
        - context: The primary context (personal/work/research)
        - next_actions: List of action items or todos
        - key_concepts: List of key concepts mentioned
        
        Return only valid JSON without any other text.
        
        Content to analyze:
        {content}
        """)
        
        response = self.llm.invoke(prompt.format(content=content[:1500]))
        
        try:
            # Try to find JSON in the response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                metadata_dict = json.loads(json_str)
                
                return MarkdownMetadata(
                    summary=metadata_dict.get("summary", ""),
                    context=metadata_dict.get("context", ""),
                    next_actions=metadata_dict.get("next_actions", []),
                    key_concepts=metadata_dict.get("key_concepts", []),
                    last_processed=datetime.now(),
                    word_count=len(content.split()),
                    links=self._extract_wiki_links(content),
                    status="active"
                )
            else:
                raise ValueError("No JSON found in response")
            
        except (json.JSONDecodeError, ValueError) as e:
            if self.verbose:
                click.echo(f"Error parsing LLM response: {e}")
                click.echo(f"Raw response: {response}")
            # Return default metadata if parsing fails
            return MarkdownMetadata(
                summary="Error processing content",
                context="unknown",
                next_actions=[],
                key_concepts=[],
                last_processed=datetime.now(),
                word_count=len(content.split()),
                links=self._extract_wiki_links(content),
                status="error"
            )
    
    def _extract_wiki_links(self, content: str) -> List[str]:
        """Extract all [[wiki-style]] links from content."""
        pattern = r'\[\[(.*?)\]\]'
        matches = re.findall(pattern, content)
        return [match.strip() for match in matches]
    
    def process_file(self, file_path: Path) -> None:
        """Process a markdown file and update its metadata."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Read existing frontmatter
        post = frontmatter.load(file_path)
        
        # Check if file should be processed
        if not self.should_process_file(post):
            return
            
        content = post.content
        
        # Extract new metadata
        metadata = self.extract_metadata(content)
        
        # Update frontmatter
        post.metadata.update({
            "summary": metadata.summary,
            "context": metadata.context,
            "next_actions": metadata.next_actions,
            "key_concepts": metadata.key_concepts,
            "last_processed": metadata.last_processed.isoformat(),
            "word_count": metadata.word_count,
            "links": metadata.links,
            "status": metadata.status
        })
        
        # Write back to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(frontmatter.dumps(post))

@click.command()
@click.argument('path', type=click.Path(exists=True))
@click.option('--recursive/--no-recursive', default=False, help='Process files recursively')
@click.option('--model', default='deepseek-r1:8b', help='Ollama model to use')
@click.option('--verbose', '-v', is_flag=True, help='Show processing details')
@click.option('--skip-recent', type=int, help='Skip files processed within N days')
def main(path: str, recursive: bool, model: str, verbose: bool, skip_recent: Optional[int]):
    """Process markdown files and add metadata."""
    processor = MarkdownProcessor(
        model_name=model, 
        verbose=verbose,
        skip_recent_days=skip_recent
    )
    path = Path(path)
    
    if verbose:
        click.echo(f"Processing {'recursively' if recursive else 'non-recursively'} from {path}")
        if skip_recent:
            click.echo(f"Skipping files processed within {skip_recent} days")
    
    if path.is_file() and path.suffix == '.md':
        if verbose:
            click.echo(f"Processing file: {path}")
        processor.process_file(path)
    elif path.is_dir():
        pattern = '**/*.md' if recursive else '*.md'
        for file_path in path.glob(pattern):
            if verbose:
                click.echo(f"Processing file: {file_path}")
            processor.process_file(file_path)

if __name__ == '__main__':
    main() 
