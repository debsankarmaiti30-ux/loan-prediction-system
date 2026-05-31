import streamlit_authenticator as stauth

hashed_password = stauth.Hasher.hash("admin123")

print(hashed_password)