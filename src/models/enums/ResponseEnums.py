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
