"""Authentication endpoints: register, login, current user."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from backend.core.config import settings
from backend.core.database import get_db
from backend.core.security import create_access_token, get_current_user
from backend.models.user import User, UserRole
from backend.schemas.auth import RegisterRequest, TokenResponse, UserOut
from backend.services import jurisdiction as juris_svc
from backend.services import users as user_svc

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post(
    "/register",
    response_model=UserOut,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new user account",
)
def register(payload: RegisterRequest, db: Session = Depends(get_db)) -> User:
    if user_svc.get_user_by_email(db, payload.email) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="Email already registered"
        )

    role = payload.role if settings.ALLOW_REGISTRATION_ROLE_SELECT else UserRole.PILOT

    jurisdiction_ids = list(payload.jurisdiction_ids)
    if payload.jurisdiction_codes:
        jurisdiction_ids += juris_svc.resolve_codes_to_ids(db, payload.jurisdiction_codes)

    return user_svc.create_user(
        db,
        name=payload.name,
        email=payload.email,
        password=payload.password,
        role=role,
        phone_number=payload.phone_number,
        jurisdiction_ids=sorted(set(jurisdiction_ids)),
    )


@router.post("/login", response_model=TokenResponse, summary="Exchange credentials for a JWT")
def login(
    form: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
) -> TokenResponse:
    # OAuth2 form uses "username"; we treat it as the email.
    user = user_svc.authenticate(db, form.username, form.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="User account is disabled"
        )

    token = create_access_token(user.id, extra_claims={"role": user.role.value})
    return TokenResponse(
        access_token=token,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=UserOut.model_validate(user),
    )


@router.get("/me", response_model=UserOut, summary="The authenticated user")
def me(current_user: User = Depends(get_current_user)) -> User:
    return current_user
