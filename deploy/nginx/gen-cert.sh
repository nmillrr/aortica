#!/usr/bin/env bash
# gen-cert.sh — generate a self-signed TLS certificate for local prod testing.
#
# Idempotent: does nothing if a certificate already exists. The generated
# certificate is NOT for production use — it exists only so `make prod` can
# demonstrate TLS termination on localhost.
set -euo pipefail

CERT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/certs"
CRT="${CERT_DIR}/aortica.crt"
KEY="${CERT_DIR}/aortica.key"

mkdir -p "${CERT_DIR}"

if [[ -f "${CRT}" && -f "${KEY}" ]]; then
    echo "TLS certificate already present at ${CERT_DIR} — skipping."
    exit 0
fi

echo "Generating self-signed TLS certificate in ${CERT_DIR} ..."
openssl req -x509 -nodes -newkey rsa:2048 \
    -keyout "${KEY}" \
    -out "${CRT}" \
    -days 365 \
    -subj "/C=US/ST=Local/L=Local/O=Aortica/CN=localhost" \
    -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"

chmod 600 "${KEY}"
echo "Done. Self-signed certificate valid for 365 days (localhost only)."
