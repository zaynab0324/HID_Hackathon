# Deploying the HID Liveness Kiosk on AWS (EC2 + IAM role + CloudFront HTTPS)

Browsers only allow the camera on **HTTPS**. CloudFront gives HTTPS (`https://xxxx.cloudfront.net`)
with no certificate setup. The EC2 machine gets AWS access from an **IAM role**, so there are
**no access keys on the server**.

## 1. IAM role for the server (console, about 3 min)
IAM → Roles → **Create role** → Trusted entity: **AWS service → EC2** → attach:
- `AmazonRekognitionFullAccess`
- `AmazonBedrockFullAccess`
Name: `HIDKioskServerRole` → Create.
Then open the role → **Add permissions → Create inline policy → JSON**:
```json
{"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Action": "sts:AssumeRole",
  "Resource": "arn:aws:iam::*:role/HIDKioskLivenessRole"}]}
```
(Lets the server hand out the liveness-only browser credentials.) Name it `AssumeKioskRole`.

## 2. Launch the EC2 machine (about 5 min)
EC2 → **Launch instance**
- Name `hid-kiosk` · **Amazon Linux 2023** · type **t3.small**
- Key pair: **Proceed without a key pair** (we use EC2 Instance Connect in the browser)
- Network: allow **SSH** from *My IP*; **Add security group rule**: Custom TCP **8000** and **8501**, source *Anywhere* (CloudFront connects here)
- Advanced details → **IAM instance profile: HIDKioskServerRole**
→ **Launch**. Copy its **Public IPv4 DNS** (ec2-xx-xx-xx-xx.compute-1.amazonaws.com).

## 3. Install and start (EC2 → Connect → EC2 Instance Connect → Connect)
```bash
sudo dnf install -y python3.11 python3.11-pip git
# private repo: create a GitHub fine-grained token (read-only, this repo) and use it here
git clone https://<TOKEN>@github.com/<USER>/hid-face.git && cd hid-face
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python selftest.py                       # ALL CHECKS PASSED
aws sts get-caller-identity              # shows assumed-role/HIDKioskServerRole
export HOST=0.0.0.0 ADMIN_PASSWORD='choose-a-strong-one' GUARD_PIN='choose-a-pin'
nohup python server.py > kiosk.log 2>&1 &
nohup streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true > admin.log 2>&1 &
```

## 4. HTTPS with CloudFront (about 10 min; it takes 5-10 min to go live)
CloudFront → **Create distribution** (do this twice: kiosk = port 8000, admin = port 8501)
- Origin domain: the **EC2 Public IPv4 DNS** · Protocol: **HTTP only** · HTTP port: **8000** (second one: **8501**)
- Viewer protocol policy: **Redirect HTTP to HTTPS**
- Allowed HTTP methods: **GET, HEAD, OPTIONS, PUT, POST, PATCH, DELETE**
- Cache policy: **CachingDisabled** · Origin request policy: **AllViewer**
- Web Application Firewall: *Do not enable* (or enable for extra security)
→ Create. Open `https://<distribution>.cloudfront.net` → kiosk (and the admin console on the second one).

## 5. One-time setup in the deployed admin console
- **Security tab → Create / update kiosk role**, then **Test kiosk credentials** (should say *least-privilege*)
- Enroll people, add schedules, create offline codes (data lives on the server now)

## Production next steps (architecture slide)
API Gateway + Lambda for the decision API · DynamoDB (people, schedule, hash-chained log) encrypted with KMS ·
Cognito for kiosk-device and staff sign-in · SNS alerts to guards · CloudWatch dashboards (FAR/FRR, latency) ·
AWS WAF on CloudFront · encrypted local cache on each kiosk for offline mode.
