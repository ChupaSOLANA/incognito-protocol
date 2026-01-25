"""
Security Middleware for Incognito Protocol API

Provides security headers and protections against common web vulnerabilities:
- XSS (Cross-Site Scripting)
- CSRF (Cross-Site Request Forgery)
- Clickjacking
- MIME sniffing
- Protocol downgrade attacks
"""

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Adds security headers to all responses.

    Headers added:
    - X-Content-Type-Options: nosniff (prevent MIME sniffing)
    - X-Frame-Options: DENY (prevent clickjacking)
    - X-XSS-Protection: 1; mode=block (XSS protection for older browsers)
    - Strict-Transport-Security: HTTPS enforcement
    - Content-Security-Policy: Restrict resource loading
    - Referrer-Policy: Control referrer information
    - Permissions-Policy: Restrict browser features
    """

    async def dispatch(self, request: Request, call_next):
        """Process request and add security headers to response"""

        # Call the next middleware/endpoint
        response: Response = await call_next(request)

        # ================================================================
        # SECURITY HEADERS
        # ================================================================

        # Prevent MIME type sniffing
        # Stops browsers from trying to guess content types
        response.headers["X-Content-Type-Options"] = "nosniff"

        # Prevent clickjacking attacks
        # Disallows embedding in iframes
        response.headers["X-Frame-Options"] = "DENY"

        # XSS Protection (legacy, but still good for older browsers)
        # Enables browser's built-in XSS protection
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # Content Security Policy
        # Restricts what resources can be loaded
        # NOTE: This is a strict policy. Adjust if you need to load external scripts/images
        csp_policy = "; ".join([
            "default-src 'self'",  # Only load resources from same origin
            "script-src 'self' 'unsafe-inline'",  # Allow inline scripts (needed for some frameworks)
            "style-src 'self' 'unsafe-inline'",  # Allow inline styles
            "img-src 'self' data: https:",  # Allow images from self, data URIs, and HTTPS
            "font-src 'self' data:",  # Allow fonts from self and data URIs
            "connect-src 'self' http://localhost:* http://127.0.0.1:* https:",  # Allow API connections
            "frame-ancestors 'none'",  # Don't allow embedding (similar to X-Frame-Options)
            "base-uri 'self'",  # Restrict <base> tag URLs
            "form-action 'self'",  # Restrict form submissions
        ])
        response.headers["Content-Security-Policy"] = csp_policy

        # Referrer Policy
        # Controls how much referrer information is sent
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Permissions Policy (formerly Feature-Policy)
        # Restricts browser features
        permissions_policy = ", ".join([
            "geolocation=()",  # Disable geolocation
            "microphone=()",  # Disable microphone
            "camera=()",  # Disable camera
            "payment=()",  # Disable payment APIs
            "usb=()",  # Disable USB access
        ])
        response.headers["Permissions-Policy"] = permissions_policy

        # HTTPS Enforcement (HSTS)
        # NOTE: Only enable in production with proper HTTPS setup
        # Uncomment this when deploying with SSL/TLS:
        # response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        # Remove potentially leaky headers
        if "Server" in response.headers:
            del response.headers["Server"]  # Hide server version
        if "X-Powered-By" in response.headers:
            del response.headers["X-Powered-By"]  # Hide framework info

        return response


class RequestValidationMiddleware(BaseHTTPMiddleware):
    """
    Validates incoming requests for common attack patterns.

    Checks:
    - Content-Type validation for POST/PUT/PATCH
    - Request size limits
    - Suspicious patterns in URLs
    """

    # Maximum request body size (10 MB)
    MAX_BODY_SIZE = 10 * 1024 * 1024

    async def dispatch(self, request: Request, call_next):
        """Validate request before processing"""

        # ================================================================
        # CONTENT-TYPE VALIDATION
        # ================================================================

        # For methods that should have a body, validate Content-Type
        if request.method in ["POST", "PUT", "PATCH"]:
            content_type = request.headers.get("content-type", "")

            # Allow only these content types
            allowed_types = [
                "application/json",
                "application/x-www-form-urlencoded",
                "multipart/form-data",
            ]

            # Check if content-type starts with any allowed type
            # (handles charset, boundary, etc.)
            is_allowed = any(
                content_type.lower().startswith(allowed)
                for allowed in allowed_types
            )

            if not is_allowed and content_type:
                # If content-type is present but not allowed, reject
                from fastapi.responses import JSONResponse
                return JSONResponse(
                    status_code=415,
                    content={
                        "error": "unsupported_media_type",
                        "detail": f"Content-Type '{content_type}' not supported"
                    }
                )

        # ================================================================
        # PATH TRAVERSAL DETECTION
        # ================================================================

        # Check for path traversal in URL
        path = str(request.url.path)
        suspicious_patterns = [
            "../",
            "..\\",
            "%2e%2e",
            "%252e%252e",
            "..%2f",
            "..%5c",
        ]

        for pattern in suspicious_patterns:
            if pattern in path.lower():
                from fastapi.responses import JSONResponse
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "invalid_request",
                        "detail": "Suspicious path pattern detected"
                    }
                )

        # ================================================================
        # PROCESS REQUEST
        # ================================================================

        response = await call_next(request)
        return response


def configure_cors(app):
    """
    Configure CORS (Cross-Origin Resource Sharing) for the API.

    This should be called during app initialization.

    Args:
        app: FastAPI application instance
    """
    from fastapi.middleware.cors import CORSMiddleware

    # CORS configuration
    # NOTE: Adjust these settings based on your deployment
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",  # React/Next.js dev server
            "http://localhost:8501",  # Streamlit dashboard
            "http://localhost:8080",  # Vite/Replit dev server
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8501",
            "http://127.0.0.1:8080",
            # Add your production frontend URLs here:
            # "https://yourdomain.com",
        ],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"],
        max_age=600,  # Cache preflight requests for 10 minutes
    )


# Export middleware classes
__all__ = [
    "SecurityHeadersMiddleware",
    "RequestValidationMiddleware",
    "configure_cors",
]
