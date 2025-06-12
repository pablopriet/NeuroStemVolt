from processing.base import Processor  # Use relative import
import inspect

class PipelineManager:
    def __init__(self, list_processors=None):
        self.list_processors = list_processors or []

    def add_processor(self, processor):
        self.list_processors.append(processor)

    def run(self, spheroid_file):
        data = spheroid_file.get_data()
        context = {}  # Initialize context
        for processor in self.list_processors:
            # Check if the processor's process method accepts a 'context' argument, if so, pass it
            # Otherwise, call it without context, context is just a way to pass metadata or additional info
            # To the SpheroidFile class
            process_signature = inspect.signature(processor.process)
            if 'context' in process_signature.parameters:
                data = processor.process(data, context=context)
            else:
                data = processor.process(data)
        spheroid_file.processed_data = data
        spheroid_file.update_metadata(context)  # Update metadata with context


