"""Python implementations of the callable tools and their schemas."""
# Encapsulate the functions implemented in `memory.py` as "tools" for LLM to call

from typing import Any, Dict, List, Optional, Type, Union
from dataclasses import dataclass
from memory import Memory


@dataclass
class Parameter:
    """Represents a function parameter with its type and requirements."""
    name: str
    type: str
    description: str
    required: bool = True
    enum: Optional[List[str]] = None # Optional list of values (limiting the parameter values)


class ToolFunction: # Provide unified interface for defining tool functions and convert the function definition to OpenAI tool Schema
    """Base class for defining tool functions in a human-friendly way."""
    
    name: str
    description: str
    parameters: List[Parameter]
    
    @classmethod
    def execute(cls, memory: Memory, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the function with the given arguments."""
        raise NotImplementedError("Subclasses must implement execute()")
    
    @classmethod
    def to_schema(cls, memory: Memory = None) -> Dict[str, Any]:
        """Convert the function definition to OpenAI tool schema format.
        
        Args:
            memory: Memory instance to get configuration from (e.g., including_core)
        """
        properties = {}
        required = []
        
        for param in cls.parameters:
            param_schema = {
                "type": param.type,
                "description": param.description
            }
            
            # Handle dynamic enum generation based on memory configuration
            if param.enum:
                enum_values = param.enum.copy()
                # If this is a memory_type parameter and memory is provided, filter based on including_core
                if param.name == "memory_type" and memory is not None:
                    if not memory.including_core and "core" in enum_values:
                        enum_values.remove("core")
                param_schema["enum"] = enum_values
            
            properties[param.name] = param_schema
            if param.required:
                required.append(param.name)
        
        return {
            "type": "function",
            "function": {
                "name": cls.name, # e.g., "new_memory_insert"
                "description": cls.description, # e.g., "Infer a new memory and append it to a memory store."
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


# Example function definitions
class NewMemoryInsert(ToolFunction):
    name = "new_memory_insert"
    description = "Infer a new memory and append it to a memory store. Creates a new memory item with a unique ID. Note: Core memory cannot be inserted, only updated."
    parameters = [
        Parameter(
            name="memory_type",
            type="string",
            description="Type of memory to insert: 'semantic' (general knowledge) or 'episodic' (specific experiences). Core memory cannot be inserted.",
            enum=["semantic_memory", "episodic_memory"]
        ),
        Parameter(
            name="content",
            type="string",
            description="Content of the memory to insert. Creates a new memory item with a unique ID."
        ),
    ]
    
    @classmethod
    def execute(cls, memory: Memory, arguments: Dict[str, Any]) -> Dict[str, Any]:
        assert arguments['memory_type'] in ['semantic_memory', 'episodic_memory']
        
        # Handle and debug non-string content
        content = arguments["content"]
        new_memory = memory.new_memory_insert(arguments["memory_type"].replace("_memory", ""), content)
        if new_memory is None:
            return {"status": "skipped", "message": "Memory content already exists in the memory pool, insertion skipped."}
        return {"status": "ok", "new_memory": new_memory}


class MemoryUpdate(ToolFunction):
    name = "memory_update"
    description = "Update an existing memory. For core memory, replaces the entire paragraph content. If core memory is empty, then directly write into the core memory. For semantic/episodic memories, updates the specific memory item by ID."
    parameters = [
        Parameter(
            name="memory_type",
            type="string",
            description="Type of memory to update: 'core' (simple paragraph), 'semantic' (general knowledge), or 'episodic' (specific experiences)",
            enum=["core_memory", "semantic_memory", "episodic_memory"]
        ),
        Parameter(
            name="new_content",
            type="string",
            description="New **combined** content for the memory. For core memory, this replaces the entire paragraph. For semantic/episodic, this replaces the content of the specified memory ID."
        ),
        Parameter(
            name="memory_id",
            type="string",
            description="ID of the memory to update. Required for semantic/episodic memories, ignored for core memory.",
            required=False
        ),
    ]
    
    @classmethod
    def execute(cls, memory: Memory, arguments: Dict[str, Any]) -> Dict[str, Any]:
        assert arguments['memory_type'] in ['core_memory', 'semantic_memory', 'episodic_memory']
        
        # Handle and debug non-string new_content
        new_content = arguments["new_content"]
        updated_memory = memory.memory_update(
            arguments["memory_type"].replace("_memory", ""),
            new_content,
            arguments.get("memory_id")
        )
        return {"status": "ok", "updated_memory": updated_memory}



class MemoryDelete(ToolFunction):
    name = "memory_delete"
    description = "Delete a memory. For core memory, clears the entire paragraph content. For semantic/episodic memories, deletes the specific memory item by ID."
    parameters = [
        Parameter(
            name="memory_type",
            type="string",
            description="Type of memory to delete: 'core' (simple paragraph), 'semantic' (general knowledge), or 'episodic' (specific experiences)",
            enum=["core_memory", "semantic_memory", "episodic_memory"]
        ),
        Parameter(
            name="memory_id",
            type="string",
            description="ID of the memory to delete. Required for semantic/episodic memories, ignored for core memory.",
            required=False
        ),
    ]
    
    @classmethod
    def execute(cls, memory: Memory, arguments: Dict[str, Any]) -> Dict[str, Any]:
        memory.memory_delete(arguments["memory_type"].replace("_memory", ""), arguments.get("memory_id"))
        return {"status": "ok"}


class SearchMemory(ToolFunction):
    name = "search_memory"
    description = "Search for memories using BM25 or text embedding similarity. Note that core memory is always in the system prompt, so you don't need to search it."
    parameters = [
        Parameter(
            name="memory_type",
            type="string",
            description="Type of memory to search",
            enum=["semantic_memory", "episodic_memory"]
        ),
        Parameter(
            name="query",
            type="string",
            description="Query string to search for in memory content"
        ),
        Parameter(
            name="search_method",
            type="string",
            description="Search method to use: 'bm25' for keyword-based search or 'text-embedding' for semantic similarity search",
            required=False,
            enum=["bm25", "text-embedding"]
        )
    ]
    
    @classmethod
    def execute(cls, memory: Memory, arguments: Dict[str, Any]) -> Dict[str, Any]:
        # Extract required arguments
        memory_type = arguments["memory_type"].replace("_memory", "")
        query = arguments["query"]
        
        # Extract optional arguments with defaults
        search_method = arguments.get("search_method", "bm25")
        
        results = memory.memory_search(
            memory_type=memory_type,
            query=query,
            top_k=memory.TOPK
        )
        return {"status": "ok", "results": results, "search_method": search_method}


# List of all available tool functions
MEMORY_TOOL_FUNCTIONS = [ # classmethod
    NewMemoryInsert,
    MemoryUpdate,
    MemoryDelete,
]

SEARCH_TOOL_FUNCTIONS = [
    SearchMemory,
]


# Generate the function implementations map
FUNCTION_IMPLS = { # Map classmethod to tool names
    func.name: func.execute for func in MEMORY_TOOL_FUNCTIONS + SEARCH_TOOL_FUNCTIONS
}

# Generate the OpenAI tool schemas - these functions now require a memory instance
def get_memory_tool_schemas(memory: Memory) -> List[Dict[str, Any]]:
    """Generate OpenAI tool schemas for memory functions based on memory configuration."""
    return [func.to_schema(memory) for func in MEMORY_TOOL_FUNCTIONS]

def get_search_tool_schemas(memory: Memory) -> List[Dict[str, Any]]:
    """Generate OpenAI tool schemas for search functions based on memory configuration."""
    return [func.to_schema(memory) for func in SEARCH_TOOL_FUNCTIONS]

# Backward compatibility - these will work but won't respect including_core configuration
MEMORY_TOOL_SCHEMAS = [ # Map classmethod to OpenAI tool schemas
    func.to_schema() for func in MEMORY_TOOL_FUNCTIONS
]

SEARCH_TOOL_SCHEMAS = [ # Map classmethod to OpenAI tool schemas
    func.to_schema() for func in SEARCH_TOOL_FUNCTIONS
]