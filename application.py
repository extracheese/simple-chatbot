from app import app as application

if __name__ == '__main__':
    # Run on all interfaces (0.0.0.0) and port 8000
    application.run(host='0.0.0.0', port=8000)
