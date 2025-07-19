# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
import traceback
from app.base.exceptions import HashCollisionError, CapacityError
from fastapi import APIRouter, HTTPException, Request
from app.schemas.api import CreateJobResponse
from core.common.models import IndexBuildParameters

import logging

router = APIRouter()
logger = logging.getLogger(__name__)


def _handle_build_error(
    e: Exception, index_build_parameters: IndexBuildParameters, status_code: int
):
    """
    Helper function to handle common error logging and exception raising.

    Args:
        e (Exception): The exception that was raised
        index_build_parameters (IndexBuildParameters): The parameters that were used
        status_code (int): HTTP status code to return

    Raises:
        HTTPException: With the provided status code and error message
    """
    logger.error(
        f"Build failed with error: {e}. "
        f"Index build parameters: {index_build_parameters}. "
        f"Traceback: {traceback.format_exc()}"
    )
    raise HTTPException(status_code=status_code, detail=str(e)) from e


@router.post("/_build")
def create_job(
    index_build_parameters: IndexBuildParameters, request: Request
) -> CreateJobResponse:
    """
    Create a new index build job with the provided parameters.

    This endpoint initiates a new job for building an index based on the provided
    parameters. It handles job creation through the job service and manages
    potential error scenarios.

    Args:
        index_build_parameters (IndexBuildParameters): Parameters for the index build job
        request (Request): FastAPI request object containing application state

    Returns:
        CreateJobResponse: Response object containing the created job ID

    Raises:
        HTTPException:
            - 429 status code if a hash collision occurs
            - 507 status code if system memory capacity is exceeded
    """
    try:
        print("_____________________________ index_build_parameters=", str(index_build_parameters))
        job_service = request.app.state.job_service
        job_id = job_service.create_job(index_build_parameters)
    except HashCollisionError as e:
        _handle_build_error(e, index_build_parameters, 429)
    except CapacityError as e:
        _handle_build_error(e, index_build_parameters, 507)

    return CreateJobResponse(job_id=job_id)
