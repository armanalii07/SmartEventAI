from passlib.context import CryptContext

pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto"
)

password = "Admin@123"

hashed_password = pwd_context.hash(password)

print("Password:", password)
print("Hashed Password:")
print(hashed_password)