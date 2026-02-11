import threading
import queue
import logging
import time
from rewriter_pipeline import StoryRewriterPipeline

# [FIX] Import necessary classes so we can instantiate them here
from local_llm import OllamaRewriter
from chunker import TokenAwareChunker
from stitcher import ChunkStitcher
from monitor import TemperatureGuard
from utils import load_style_prompt

# A thread-safe queue to stream logs to the UI
log_queue = queue.Queue()

class QueueHandler(logging.Handler):
    """Custom logging handler to send logs to the Web UI"""
    def emit(self, record):
        try:
            log_entry = self.format(record)
            log_queue.put(log_entry)
        except Exception:
            self.handleError(record)

class InterruptiblePipeline(StoryRewriterPipeline):
    """Subclass that allows stopping and status tracking"""
    def __init__(self, config):
        # [FIX] Do NOT call super().__init__ because it tries to load a file from disk.
        # Instead, we manually initialize the components using the config dict passed from Flask.
        
        self.config = config
        
        # Manual Component Initialization (Mirrors StoryRewriterPipeline)
        self.rewriter = OllamaRewriter(self.config)
        self.chunker = TokenAwareChunker(self.config, self.rewriter)
        self.stitcher = ChunkStitcher(self.config)
        self.temp_guard = TemperatureGuard(self.config)
        
        # Inject Guard into Rewriter
        self.rewriter.set_temp_guard(self.temp_guard)
        
        # Load Prompts
        self.style_prompt = load_style_prompt()
        self.system_prompt = self.config.get('prompts', {}).get('system_prompt', "You are an expert editor.")
        
        # Initialize Stats (Required because 'run' uses them)
        self.stats = {
            'total_chapters': 0,
            'total_chunks': 0,
            'successful_chunks': 0,
            'failed_chunks': 0,
            'total_time': 0,
            'validation_failures': 0
        }
        
        # Control flags
        self.stop_requested = False

    def _process_chapter(self, chapter):
        """Override to check for stop signal"""
        if self.stop_requested:
            raise InterruptedError("Stopped by user")
        return super()._process_chapter(chapter)

class PipelineThread(threading.Thread):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pipeline = None
        self.error = None
        self.running = True

    def run(self):
        # Setup Logging to Queue
        root_logger = logging.getLogger()
        queue_handler = QueueHandler()
        queue_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        root_logger.addHandler(queue_handler)
        
        try:
            self.pipeline = InterruptiblePipeline(self.config)
            self.pipeline.run()
        except InterruptedError:
            logging.info("Pipeline stopped by user.")
        except Exception as e:
            self.error = str(e)
            logging.error(f"Pipeline Error: {e}")
        finally:
            self.running = False
            root_logger.removeHandler(queue_handler)

    def stop(self):
        if self.pipeline:
            self.pipeline.stop_requested = True