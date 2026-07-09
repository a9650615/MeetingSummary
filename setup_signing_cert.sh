#!/usr/bin/env bash
# One-time: create a self-signed "Code Signing" identity so macOS TCC grants
# (Screen & System Audio Recording, Microphone) SURVIVE rebuilds. TCC keys a
# grant to the app's code identity; adhoc builds get a fresh cdhash every time
# and lose the grant. A stable self-signed identity fixes that — no Apple
# account, no notarization (Gatekeeper still warns on OTHER machines, which is
# fine for dev + self-distributed builds).
#
#   ./setup_signing_cert.sh          # create "MeetingSummary Dev" in login keychain
#   FLOATPANEL_SIGN_ID=... override the name
#
# Idempotent: skips creation if the identity already exists.
set -eu
NAME="${FLOATPANEL_SIGN_ID:-MeetingSummary Dev}"
KEYCHAIN="$HOME/Library/Keychains/login.keychain-db"
# find-identity WITHOUT -v: a self-signed cert is untrusted (CSSMERR_TP_NOT_TRUSTED),
# so it never shows under "valid identities only" — but codesign can still USE it,
# and that's all we need (TCC keys on the designated requirement = identifier +
# cert hash, which is stable across rebuilds regardless of trust).
if security find-identity -p codesigning 2>/dev/null | grep -q "$NAME"; then
  echo "signing identity already present: $NAME"
  exit 0
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
cat > "$TMP/cert.conf" <<EOF
[req]
distinguished_name = dn
x509_extensions = ext
prompt = no
[dn]
CN = $NAME
[ext]
basicConstraints = critical,CA:false
keyUsage = critical,digitalSignature
extendedKeyUsage = critical,codeSigning
EOF

# Self-signed code-signing cert (10 yr), key unencrypted, then pack to p12.
# -legacy + a non-empty password: OpenSSL 3's default p12 MAC/cipher is rejected
# by macOS `security` ("MAC verification failed"); the legacy algo set imports fine.
openssl req -x509 -newkey rsa:2048 -sha256 -days 3650 -nodes \
  -keyout "$TMP/key.pem" -out "$TMP/cert.pem" -config "$TMP/cert.conf"
openssl pkcs12 -export -legacy -inkey "$TMP/key.pem" -in "$TMP/cert.pem" \
  -out "$TMP/id.p12" -passout pass:msdev

# Import cert+key into the login keychain and pre-authorize codesign to use the
# private key (so signing doesn't pop a keychain prompt each time — you may still
# get ONE "Always Allow" dialog the first time you sign; click it once).
security import "$TMP/id.p12" -k "$KEYCHAIN" -P msdev -T /usr/bin/codesign -A

echo
if security find-identity -p codesigning 2>/dev/null | grep -q "$NAME"; then
  echo "created signing identity: $NAME (self-signed; shows CSSMERR_TP_NOT_TRUSTED — that's fine, codesign still uses it)"
  echo "next: ./build_floatpanel_app.sh  (now signs with it; TCC grant persists across rebuilds)"
else
  echo "warning: identity not visible to codesigning — check Keychain Access" >&2
  exit 1
fi
