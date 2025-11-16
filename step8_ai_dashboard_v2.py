import streamlit_authenticator as stauth

# Define usernames, names and passwords (prefer hashed)
names = ['Admin User', 'Normal User']
usernames = ['Vaishnavi', 'user']
passwords = ['#bharat@123', '123']

# Hash passwords for security
hashed_passwords = stauth.Hasher(passwords).generate()

# Create authenticator instance
authenticator = stauth.Authenticate(
    names, usernames, hashed_passwords,
    'cookie_name', 'signature_key', cookie_expiry_days=1
)

# Show login widget
name, auth_status, username = authenticator.login('Login', 'main')

if auth_status:
    authenticator.logout('Logout', 'main')
    st.write(f'Welcome *{name}*')
    # Your app logic here
elif auth_status is False:
    st.error('Username or password is incorrect')
else:
    st.warning('Please enter your username and password')
