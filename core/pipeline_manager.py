from core.processing.base import Processor  # Use relative import
import inspect

class PipelineManager:
    """
    Manage and execute a sequence of processing steps on spheroid data files.

    This class maintains an ordered list of processing steps (processors) and applies them
    sequentially to a spheroid file's data. Each processor must implement a `process` method.

    Args:
        list_processors (list, optional): A list of Processor instances to run in sequence.
    """
    def __init__(self, list_processors=None):
        """
        Initialize the pipeline manager with an optional list of processors.

        Args:
            list_processors (list, optional): Initial list of Processor instances.

        Returns:
            None
        """
        self.list_processors = list_processors or []

    def add_processor(self, processor):
        """
        Add a processor to the processing pipeline.

        Args:
            processor (Processor): A processor instance that implements a `process` method.

        Returns:
            None
        """
        self.list_processors.append(processor)

    def run(self, spheroid_file, context=None):
        """
        Execute the processing pipeline on a single spheroid file.

        Each processor is applied sequentially to the file's processed data.
        If a processor accepts a `context` argument, it is passed in.

        Args:
            spheroid_file: An object containing spheroid data and metadata.
            context (dict, optional): Shared state between processors.

        Returns:
            None
        """
        data = spheroid_file.get_processed_data()
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


