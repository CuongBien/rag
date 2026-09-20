import os
import sys
from dotenv import load_dotenv
from neo4j import GraphDatabase

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dotenv_path = os.path.join(project_root, ".env")
    load_dotenv(dotenv_path, override=True)

    uri = os.getenv("NEO4J_URI", "").strip()
    username = os.getenv("NEO4J_USERNAME", "neo4j").strip()
    password = os.getenv("NEO4J_PASSWORD", "").strip()
    database = os.getenv("NEO4J_DATABASE", "neo4j").strip()

    print("=" * 60)
    print("🔍 NEO4J CLOUD (AURA) CONNECTIVITY CHECK")
    print("=" * 60)
    print(f"URI      : {uri}")
    print(f"Username : {username}")
    print(f"Database : {database}")
    print(f"Password : {'*' * len(password) if password else '[NOT SET]'}")
    print("-" * 60)

    if not uri or "<your-instance-id>" in uri:
        print("❌ Lỗi: Bạn chưa điền NEO4J_URI chính xác trong file .env!")
        print("👉 Định dạng đúng của Neo4j Aura: neo4j+s://<dbid>.databases.neo4j.io")
        sys.exit(1)

    if not uri.startswith("neo4j+s://") and not uri.startswith("bolt+s://"):
        print("⚠️ Cảnh báo: Neo4j Aura Cloud yêu cầu kết nối mã hóa TLS (neo4j+s:// hoặc bolt+s://).")

    if not password or "<your-aura-password>" in password:
        print("❌ Lỗi: Bạn chưa điền NEO4J_PASSWORD trong file .env!")
        sys.exit(1)

    try:
        print("⏳ Đang kết nối tới Neo4j Aura...")
        driver = GraphDatabase.driver(uri, auth=(username, password))
        driver.verify_connectivity()
        print("✅ Kết nối Driver Neo4j thành công!")

        with driver.session(database=database) as session:
            result = session.run("MATCH (n) RETURN count(n) AS node_count")
            record = result.single()
            count = record["node_count"] if record else 0
            print(f"📊 Trạng thái Database '{database}': Đang có {count} node(s).")

        driver.close()
        print("🎉 Mọi thiết lập Neo4j Cloud đều chính xác và sẵn sàng hoạt động!")
    except Exception as e:
        print(f"\n❌ Lỗi kết nối Neo4j: {e}")
        print("Gợi ý kiểm tra:")
        print(" 1. Instance trên Neo4j Aura có đang ở trạng thái 'Running' (không bị Paused) không?")
        print(" 2. Password có đúng với file credentials lúc tạo database không?")
        print(" 3. URI có đúng dạng neo4j+s://<dbid>.databases.neo4j.io không?")
        sys.exit(1)

if __name__ == "__main__":
    main()
