# Bhitt AI Backend - Production Deployment

## Clean Production Structure
```
├── .env                          # Environment variables (configure for prod)
├── .gitignore                    # Git ignore file
├── requirements.txt              # Python dependencies
├── run.py                       # Main application entry point
├── app/                         # Core application
│   ├── __init__.py
│   ├── api/                     # API endpoints
│   ├── config/                  # Configuration
│   ├── models/                  # AI models (chatbot, poetry)
│   ├── services/                # Business logic
│   └── utils/                   # Utilities
├── deploy/                      # Deployment configurations
│   ├── bhitt-ai-backend.nginx.conf
│   └── bhitt-ai-backend.service
└── logs/                        # Application logs (empty)
```

## Pre-deployment Checklist

### 1. Environment Configuration
- [ ] Update `.env` with production database credentials
- [ ] Set production API keys and secrets
- [ ] Configure production URLs and ports

### 2. Dependencies
```bash
pip install -r requirements.txt
```

### 3. Database Setup
- [ ] Create production MySQL database
- [ ] Run database migrations
- [ ] Test database connectivity

### 4. Deployment Files
- [ ] Update `deploy/bhitt-ai-backend.nginx.conf` with production domain
- [ ] Update `deploy/bhitt-ai-backend.service` with production paths
- [ ] Configure SSL certificates

### 5. Security
- [ ] Change default passwords
- [ ] Update JWT secret keys
- [ ] Configure firewall rules
- [ ] Enable HTTPS only

## Deployment Commands

### Start Application
```bash
python run.py
```

### System Service (Ubuntu/CentOS)
```bash
sudo cp deploy/bhitt-ai-backend.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable bhitt-ai-backend
sudo systemctl start bhitt-ai-backend
```

### Nginx Configuration
```bash
sudo cp deploy/bhitt-ai-backend.nginx.conf /etc/nginx/sites-available/
sudo ln -s /etc/nginx/sites-available/bhitt-ai-backend.nginx.conf /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

## Monitoring
- Application logs: `logs/app.log`
- System service: `sudo journalctl -u bhitt-ai-backend -f`
- Nginx logs: `/var/log/nginx/`

## Features Included
✅ Session-based conversation management
✅ JWT authentication
✅ Poetry search engine
✅ General chatbot with RAG
✅ User management and subscriptions
✅ Rate limiting
✅ Database integration
✅ Cross-model session support

## API Endpoints
- Authentication: `/api/login`, `/api/register`
- User: `/api/user-details`, `/api/subscription-details`
- Chatbot: `/api/chatbot` (POST)
- Poetry: `/api/poetry` (POST)
- Sessions: `/api/sessions/*`

All unnecessary development files have been removed for production.
