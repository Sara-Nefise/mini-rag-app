from enum import Enum

class ResponseSignal(Enum):
  
  FILE_VALIDATED_SUCCESSFULLY = "file_validated_successfully"
  FILE_VALIDATED_FAILED = "file_validation_failed"

  FILE_TYPE_NOT_SUPPORTED = "file_type_not_supported"
  FILE_SIZE_EXCEEDS = "file_size_exceeds_the_maximum_allowed"
  FILE_UPLOADED_SUCCESSFULLY = "file_uploaded_successfully"
  FILE_UPLOADED_FAILED = "file_upload_failed"
  PROCESSING_FAILED = "processing_failed"
  PROCESSING_SUCCESSFUL = "processing_successful"
  NO_FILES_ERROR= "no_files_error"
  FILE_ID_ERROR= "no_file_found_with_this_id"
  PROJECT_NOT_FOUND_ERROR= "project_not_found_error"
  INSERT_INTO_VECTORDB_ERROR = "insert_into_vectordb_error"
  INSERT_INTO_VECTORDB_SUCCESS ="insert_into_vectordb_success"
  VECTORDB_COLLECTION_RETRIEVED ="vectordo_collection_retrieved"
  VECTORDB_SEARCH_ERROR = "vectordb_search_error"
  VECTORDB_SEARCH_SUCCESS ="vectordb_search_success"
  RAG_ANSWER_ERROR="rag_answer_error"
  RAG_ANSWER_SUCCESS="rag_answer_success"
