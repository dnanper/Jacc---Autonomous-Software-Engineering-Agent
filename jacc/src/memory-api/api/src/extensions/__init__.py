"""
Hindsight Extensions System.

Extensions allow customizing and extending Hindsight behavior without modifying core code.
Extensions are loaded via environment variables pointing to implementation classes.

Example:
    HINDSIGHT_API_OPERATION_VALIDATOR_EXTENSION=mypackage.validators:MyValidator
    HINDSIGHT_API_OPERATION_VALIDATOR_MAX_RETRIES=3

    HINDSIGHT_API_HTTP_EXTENSION=mypackage.http:MyHttpExtension
    HINDSIGHT_API_HTTP_SOME_CONFIG=value

Extensions receive an ExtensionContext that provides a controlled API for interacting
with the system (e.g., running migrations for tenant schemas).
"""

from src.extensions.base import Extension
from src.extensions.builtin import ApiKeyTenantExtension
from src.extensions.context import DefaultExtensionContext, ExtensionContext
from src.extensions.http import HttpExtension
from src.extensions.loader import load_extension
from src.extensions.operation_validator import (
    OperationValidationError,
    OperationValidatorExtension,
    RecallContext,
    RecallResult,
    ReflectContext,
    ReflectResultContext,
    RetainContext,
    RetainResult,
    ValidationResult,
)
from src.extensions.tenant import (
    AuthenticationError,
    TenantContext,
    TenantExtension,
)
from src.models import RequestContext

__all__ = [
    # Base
    "Extension",
    "load_extension",
    # Context
    "ExtensionContext",
    "DefaultExtensionContext",
    # HTTP Extension
    "HttpExtension",
    # Operation Validator
    "OperationValidationError",
    "OperationValidatorExtension",
    "RecallContext",
    "RecallResult",
    "ReflectContext",
    "ReflectResultContext",
    "RetainContext",
    "RetainResult",
    "ValidationResult",
    # Tenant/Auth
    "ApiKeyTenantExtension",
    "AuthenticationError",
    "RequestContext",
    "TenantContext",
    "TenantExtension",
]
