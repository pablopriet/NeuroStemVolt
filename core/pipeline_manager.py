from processing.base import Processor  # Use relative import
import inspect

class PipelineManager:
    """
    Usage:
        A class to manage a pipeline of processors for processing spheroid files.
        It allows adding processors dynamically and running them sequentially on spheroid files.
    Inputs:
    - list_processors: A list of Processor instances to be executed in sequence.
    
    """
    def __init__(self, list_processors=None):
        self.list_processors = list_processors or []

    def add_processor(self, processor):
        """
        Usage:
            Adds a processor to the pipeline.
        """
        self.list_processors.append(processor)

    def run(self, spheroid_file, context=None):
        """
        Runs the processing pipeline for a single spheroid file.
        """
        data = spheroid_file.get_data()
        context = context or {}  # Initialize context if not provided
        for processor in self.list_processors:
            # Check if the processor's process method accepts a 'context' argument
            process_signature = inspect.signature(processor.process)
            if 'context' in process_signature.parameters:
                data = processor.process(data, context=context)
            else:
                data = processor.process(data)
        spheroid_file.processed_data = data
        spheroid_file.update_metadata(context)  # Update metadata with context


