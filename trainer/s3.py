"""DRfC 의 저장소(MinIO, S3 호환)에서 객체를 읽는 최소한의 S3 클라이언트 (표준 라이브러리만 사용, AWS Signature V4).

왜 파일을 직접 읽지 않고 S3 API 를 쓰는가: MinIO 는 객체를 평범한 파일이 아니라 `<키>/xl.meta` 같은 디렉터리 구조로 저장한다
(data/minio/bucket/<모델>/metrics/TrainingMetrics.json 이 파일이 아니라 디렉터리다). 내용은 S3 API 로만 제대로 읽을 수 있다.
"""
import configparser
import datetime
import hashlib
import hmac
import http.client
import os
import urllib.parse
import xml.etree.ElementTree as ET

_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


class S3Error(Exception):
    pass


def _h(key, msg):
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


class S3Client:
    def __init__(self, endpoint, access_key, secret_key, region="us-east-1", timeout=6):
        u = urllib.parse.urlparse(endpoint)
        if u.scheme not in ("http", "https") or not u.hostname:
            raise S3Error(f"저장소 주소가 올바르지 않습니다: {endpoint}")
        self.scheme, self.host, self.port = u.scheme, u.hostname, u.port or (443 if u.scheme == "https" else 80)
        self.netloc = u.netloc
        self.access_key, self.secret_key, self.region, self.timeout = access_key, secret_key, region, timeout

    @classmethod
    def from_drfc(cls, sysenv, aws_dir=None):
        """DRfC 의 system.env 값과 ~/.aws/credentials 의 프로필(기본 minio)로 만든다. 자격 증명이 없으면 None."""
        profile = sysenv.get("DR_LOCAL_S3_PROFILE", "minio")
        cp = configparser.ConfigParser()
        cp.read(os.path.join(aws_dir or os.path.expanduser("~/.aws"), "credentials"))
        if profile not in cp or "aws_access_key_id" not in cp[profile]:
            return None
        endpoint = sysenv.get("DR_REMOTE_MINIO_URL") or "http://localhost:9000"
        return cls(endpoint, cp[profile]["aws_access_key_id"], cp[profile].get("aws_secret_access_key", ""))

    # ------------------------------------------------------------------ 서명
    def _open(self, path, query=None, extra_headers=None, timeout=None):
        """서명한 GET 요청을 보내고 (연결, 응답) 을 돌려준다. 호출한 쪽이 연결을 닫아야 한다."""
        query = query or {}
        now = datetime.datetime.now(datetime.timezone.utc)
        amzdate, datestamp = now.strftime("%Y%m%dT%H%M%SZ"), now.strftime("%Y%m%d")
        headers = {"host": self.netloc, "x-amz-content-sha256": _EMPTY_SHA256, "x-amz-date": amzdate}
        headers.update({k.lower(): v for k, v in (extra_headers or {}).items()})
        names = sorted(headers)
        canon_uri = urllib.parse.quote(path, safe="/-_.~")
        canon_query = "&".join(f"{urllib.parse.quote(k, safe='-_.~')}={urllib.parse.quote(str(v), safe='-_.~')}" for k, v in sorted(query.items()))
        canon_headers = "".join(f"{k}:{headers[k].strip()}\n" for k in names)
        signed = ";".join(names)
        creq = "\n".join(["GET", canon_uri, canon_query, canon_headers, signed, _EMPTY_SHA256])
        scope = f"{datestamp}/{self.region}/s3/aws4_request"
        sts = "\n".join(["AWS4-HMAC-SHA256", amzdate, scope, hashlib.sha256(creq.encode("utf-8")).hexdigest()])
        k = _h(_h(_h(_h(("AWS4" + self.secret_key).encode("utf-8"), datestamp), self.region), "s3"), "aws4_request")
        sig = hmac.new(k, sts.encode("utf-8"), hashlib.sha256).hexdigest()
        headers["authorization"] = f"AWS4-HMAC-SHA256 Credential={self.access_key}/{scope}, SignedHeaders={signed}, Signature={sig}"
        conn_cls = http.client.HTTPSConnection if self.scheme == "https" else http.client.HTTPConnection
        conn = conn_cls(self.host, self.port, timeout=timeout or self.timeout)
        try:
            conn.request("GET", canon_uri + (("?" + canon_query) if canon_query else ""), headers={k: v for k, v in headers.items() if k != "host"})
            return conn, conn.getresponse()
        except (OSError, http.client.HTTPException) as e:
            conn.close()
            raise S3Error(f"저장소(minio)에 연결할 수 없습니다: {e}")

    def _request(self, path, query=None, extra_headers=None):
        conn, r = self._open(path, query, extra_headers)
        try:
            return r.status, r.read(), {k.lower(): v for k, v in r.getheaders()}
        except (OSError, http.client.HTTPException) as e:
            raise S3Error(f"저장소(minio)에서 읽지 못했습니다: {e}")
        finally:
            conn.close()

    # ------------------------------------------------------------------ 객체
    def get(self, bucket, key, etag=None):
        """(상태코드, 내용 bytes, etag). 없으면 404, etag 가 같으면 304(내용 없음)."""
        extra = {"if-none-match": etag} if etag else None
        status, body, hdr = self._request(f"/{bucket}/{key}", extra_headers=extra)
        if status in (200, 304, 404):
            return status, body, hdr.get("etag")
        raise S3Error(f"S3 응답 오류 {status}: {body[:160].decode('utf-8', 'replace')}")

    def open_object(self, bucket, key, range_header=None, timeout=30):
        """큰 객체(영상)를 스트리밍으로 읽는다. (상태코드, 헤더, 응답, 연결). 호출한 쪽이 연결을 닫아야 한다. range_header 는 'bytes=0-99' 같은 값."""
        extra = {"range": range_header} if range_header else None
        conn, r = self._open(f"/{bucket}/{key}", extra_headers=extra, timeout=timeout)
        return r.status, {k.lower(): v for k, v in r.getheaders()}, r, conn

    def list_objects(self, bucket, prefix="", max_pages=20):
        """접두사 아래 객체의 [{key, size, modified(epoch 초)}]."""
        out, token = [], None
        for _ in range(max_pages):
            q = {"list-type": "2", "prefix": prefix}
            if token:
                q["continuation-token"] = token
            status, body, _ = self._request(f"/{bucket}", query=q)
            if status == 404:
                return out
            if status != 200:
                raise S3Error(f"S3 목록 응답 오류 {status}: {body[:160].decode('utf-8', 'replace')}")
            root = ET.fromstring(body)
            for e in root.findall(f"{_NS}Contents"):
                try:
                    mod = datetime.datetime.fromisoformat((e.findtext(f"{_NS}LastModified") or "").replace("Z", "+00:00")).timestamp()
                except ValueError:
                    mod = None
                out.append({"key": e.findtext(f"{_NS}Key"), "size": int(e.findtext(f"{_NS}Size") or 0), "modified": mod})
            if root.findtext(f"{_NS}IsTruncated") != "true":
                break
            token = root.findtext(f"{_NS}NextContinuationToken")
            if not token:
                break
        return out

    def list(self, bucket, prefix="", delimiter=None, max_pages=5):
        """(키 목록, 하위 접두사 목록)."""
        keys, prefixes, token = [], [], None
        for _ in range(max_pages):
            q = {"list-type": "2", "prefix": prefix}
            if delimiter:
                q["delimiter"] = delimiter
            if token:
                q["continuation-token"] = token
            status, body, _ = self._request(f"/{bucket}", query=q)
            if status == 404:
                return keys, prefixes
            if status != 200:
                raise S3Error(f"S3 목록 응답 오류 {status}: {body[:160].decode('utf-8', 'replace')}")
            root = ET.fromstring(body)
            keys += [e.findtext(f"{_NS}Key") for e in root.findall(f"{_NS}Contents")]
            prefixes += [e.findtext(f"{_NS}Prefix") for e in root.findall(f"{_NS}CommonPrefixes")]
            if root.findtext(f"{_NS}IsTruncated") != "true":
                break
            token = root.findtext(f"{_NS}NextContinuationToken")
            if not token:
                break
        return keys, prefixes
