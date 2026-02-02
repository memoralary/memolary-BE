#!/bin/bash
# =============================================================================
# EC2 초기 서버 설정 스크립트
# Ubuntu 22.04 LTS 기준
# =============================================================================

set -e  # 에러 발생 시 스크립트 중단

echo "🚀 Starting EC2 server setup..."

# =============================================================================
# 1. 시스템 업데이트
# =============================================================================
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# =============================================================================
# 2. Python 3.12 설치
# =============================================================================
echo "🐍 Installing Python 3.12..."
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev python3-pip

# =============================================================================
# 3. Redis 설치
# =============================================================================
echo "🔴 Installing Redis..."
sudo apt install -y redis-server
sudo systemctl enable redis-server
sudo systemctl start redis-server

# =============================================================================
# 4. Nginx 설치
# =============================================================================
echo "🌐 Installing Nginx..."
sudo apt install -y nginx
sudo systemctl enable nginx

# =============================================================================
# 5. Git 설치 및 프로젝트 클론
# =============================================================================
echo "📥 Cloning project..."
cd ~
git clone https://github.com/memoralary/memolary-BE.git memolary-backend
cd memolary-backend

# =============================================================================
# 6. 가상환경 생성 및 의존성 설치
# =============================================================================
echo "🔧 Setting up virtual environment..."
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install gunicorn

# =============================================================================
# 7. 환경변수 파일 생성 (.env)
# =============================================================================
echo "📝 Creating .env file template..."
cat > .env << 'EOF'
# Django
SECRET_KEY=your-secret-key-here
DEBUG=False
ALLOWED_HOSTS=your-ec2-ip,your-domain.com

# OpenAI
OPENAI_API_KEY=your-openai-api-key

# Database (SQLite 사용 시 불필요)
# DATABASE_URL=postgres://user:pass@localhost/dbname

# Redis
REDIS_URL=redis://localhost:6379/0
EOF

echo "⚠️  Please edit .env file with your actual values!"

# =============================================================================
# 8. Gunicorn systemd 서비스 생성
# =============================================================================
echo "🔄 Creating Gunicorn service..."
sudo tee /etc/systemd/system/gunicorn.service > /dev/null << EOF
[Unit]
Description=Gunicorn daemon for Memolary Backend
After=network.target

[Service]
User=$USER
Group=www-data
WorkingDirectory=$HOME/memolary-backend
Environment="PATH=$HOME/memolary-backend/venv/bin"
EnvironmentFile=$HOME/memolary-backend/.env
ExecStart=$HOME/memolary-backend/venv/bin/gunicorn \\
    --workers 3 \\
    --bind unix:$HOME/memolary-backend/gunicorn.sock \\
    --access-logfile - \\
    --error-logfile - \\
    backend.wsgi:application

[Install]
WantedBy=multi-user.target
EOF

# =============================================================================
# 9. Celery systemd 서비스 생성
# =============================================================================
echo "🔄 Creating Celery service..."
sudo tee /etc/systemd/system/celery.service > /dev/null << EOF
[Unit]
Description=Celery Worker for Memolary Backend
After=network.target redis.service

[Service]
User=$USER
Group=www-data
WorkingDirectory=$HOME/memolary-backend
Environment="PATH=$HOME/memolary-backend/venv/bin"
EnvironmentFile=$HOME/memolary-backend/.env
ExecStart=$HOME/memolary-backend/venv/bin/celery -A backend worker -l info --concurrency=2

[Install]
WantedBy=multi-user.target
EOF

# =============================================================================
# 10. Nginx 설정
# =============================================================================
echo "🌐 Configuring Nginx..."
sudo tee /etc/nginx/sites-available/memolary << EOF
server {
    listen 80;
    server_name _;  # EC2 IP 또는 도메인으로 변경

    location = /favicon.ico { access_log off; log_not_found off; }
    
    location /static/ {
        root $HOME/memolary-backend;
    }

    location /media/ {
        root $HOME/memolary-backend;
    }

    location / {
        include proxy_params;
        proxy_pass http://unix:$HOME/memolary-backend/gunicorn.sock;
    }
}
EOF

sudo ln -sf /etc/nginx/sites-available/memolary /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# =============================================================================
# 11. 서비스 시작
# =============================================================================
echo "🚀 Starting services..."
sudo systemctl daemon-reload
sudo systemctl enable gunicorn celery
sudo systemctl start gunicorn celery
sudo systemctl restart nginx

# =============================================================================
# 12. 방화벽 설정 (필요시)
# =============================================================================
echo "🔥 Configuring firewall..."
sudo ufw allow 'Nginx Full'
sudo ufw allow OpenSSH
# sudo ufw enable  # 수동으로 활성화 권장

# =============================================================================
# 완료
# =============================================================================
echo ""
echo "✅ Server setup completed!"
echo ""
echo "📋 Next steps:"
echo "1. Edit ~/.memolary-backend/.env with your actual values"
echo "2. Run: python manage.py migrate"
echo "3. Run: python manage.py collectstatic"
echo "4. Restart services: sudo systemctl restart gunicorn celery nginx"
echo ""
echo "🔗 Your server should be running at: http://$(curl -s ifconfig.me)"
