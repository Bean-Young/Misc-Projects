from rdflib import Graph, Namespace, RDF, RDFS, Literal, URIRef

# 定义命名空间
EX = Namespace("http://example.org/")
SCHEMA = Namespace("http://schema.org/")

def create_knowledge_graph():
    """
    创建包含图书和作者信息的知识图谱
    
    返回:
        rdflib.Graph: 包含图书、作者及其关系的知识图谱
    """
    # 创建一个新的RDF图
    g = Graph()
    
    # 绑定命名空间前缀
    g.bind("ex", EX)
    g.bind("schema", SCHEMA)
    
    # 定义类
    g.add((EX.Book, RDF.type, RDFS.Class))
    g.add((EX.Author, RDF.type, RDFS.Class))
    
    # 定义关系
    g.add((EX.writtenBy, RDF.type, RDF.Property))
    g.add((EX.writtenBy, RDFS.domain, EX.Book))
    g.add((EX.writtenBy, RDFS.range, EX.Author))
    
    # 图书和作者数据
    books = {
        "1984": "George_Orwell",
        "Animal_Farm": "George_Orwell",
        "Harry_Potter_and_the_Philosophers_Stone": "J.K._Rowling",
        "Harry_Potter_and_the_Chamber_of_Secrets": "J.K._Rowling",
        "One_Hundred_Years_of_Solitude": "Gabriel_Garcia_Marquez"
    }
    
    # 中文名称映射
    chinese_titles = {
        "1984": "《1984》",
        "Animal_Farm": "《动物农场》",
        "Harry_Potter_and_the_Philosophers_Stone": "《哈利・波特与魔法石》",
        "Harry_Potter_and_the_Chamber_of_Secrets": "《哈利・波特与密室》",
        "One_Hundred_Years_of_Solitude": "《百年孤独》"
    }
    
    chinese_authors = {
        "George_Orwell": "乔治・奥威尔",
        "J.K._Rowling": "J.K. 罗琳",
        "Gabriel_Garcia_Marquez": "加夫列尔・加西亚・马尔克斯"
    }
    
    # 添加图书和作者到知识图谱
    for book_title, author_name in books.items():
        # 创建图书URI
        book_uri = EX[book_title]
        
        # 添加图书实例和标题
        g.add((book_uri, RDF.type, EX.Book))
        g.add((book_uri, SCHEMA.name, Literal(chinese_titles[book_title], lang="zh")))
        g.add((book_uri, SCHEMA.name, Literal(book_title.replace("_", " "), lang="en")))
        
        # 创建作者URI
        author_uri = EX[author_name]
        
        # 添加作者实例和名称
        g.add((author_uri, RDF.type, EX.Author))
        g.add((author_uri, SCHEMA.name, Literal(chinese_authors[author_name], lang="zh")))
        g.add((author_uri, SCHEMA.name, Literal(author_name.replace("_", " "), lang="en")))
        
        # 添加作者关系
        g.add((book_uri, EX.writtenBy, author_uri))
    
    return g

def query_all_books(graph):
    """
    查询知识图谱中的所有图书
    
    参数:
        graph (rdflib.Graph): 知识图谱
        
    返回:
        list: 包含所有图书名称的列表
    """
    query = """
    SELECT DISTINCT ?bookTitle
    WHERE {
        ?book a ex:Book ;
               schema:name ?bookTitle .
    }
    """
    
    results = graph.query(query, initNs={"ex": EX, "schema": SCHEMA})
    return [str(row.bookTitle) for row in results]

def query_book_authors(graph):
    """
    查询每本书及其作者
    
    参数:
        graph (rdflib.Graph): 知识图谱
        
    返回:
        list: 包含(图书, 作者)元组的列表
    """
    query = """
    SELECT ?bookTitle ?authorName
    WHERE {
        ?book a ex:Book ;
               schema:name ?bookTitle ;
               ex:writtenBy ?author .
        ?author schema:name ?authorName .
    }
    ORDER BY ?bookTitle
    """
    
    results = graph.query(query, initNs={"ex": EX, "schema": SCHEMA})
    return [(str(row.bookTitle), str(row.authorName)) for row in results]

def query_books_by_author(graph, author_name):
    """
    查询特定作者写的所有书
    
    参数:
        graph (rdflib.Graph): 知识图谱
        author_name (str): 作者名称
        
    返回:
        list: 包含图书名称的列表
    """
    query = """
    SELECT ?bookTitle
    WHERE {
        ?author a ex:Author ;
                schema:name ?authorName .
        FILTER (?authorName = "%s")
        
        ?book a ex:Book ;
              schema:name ?bookTitle ;
              ex:writtenBy ?author .
    }
    """ % author_name
    
    results = graph.query(query, initNs={"ex": EX, "schema": SCHEMA})
    return [str(row.bookTitle) for row in results]

def visualize_graph(graph):
    """
    可视化知识图谱结构
    
    参数:
        graph (rdflib.Graph): 知识图谱
    """
    print("\n知识图谱结构:")
    print("=" * 50)
    
    # 打印所有三元组
    for s, p, o in graph:
        subject = s.split("/")[-1] if isinstance(s, URIRef) else s
        predicate = p.split("/")[-1] if isinstance(p, URIRef) else p
        object_val = o.split("/")[-1] if isinstance(o, URIRef) else o
        
        print(f"{subject:50} -> {predicate:15} -> {object_val}")
    
    print("=" * 50)

def main():
    # 创建知识图谱
    kg = create_knowledge_graph()
    
    # 可视化知识图谱结构
    visualize_graph(kg)
    
    # 查询1: 获取所有图书名称
    print("\n查询1: 所有图书名称")
    print("-" * 50)
    books = query_all_books(kg)
    for i, book in enumerate(books, 1):
        print(f"{i}. {book}")
    
    # 查询2: 获取每本书的作者
    print("\n查询2: 每本书的作者")
    print("-" * 50)
    book_authors = query_book_authors(kg)
    for book, author in book_authors:
        print(f"{book} 由 {author} 所著")
    
    # 查询3: 获取乔治・奥威尔写的所有书
    print("\n查询3: 乔治・奥威尔写的所有书")
    print("-" * 50)
    orwell_books = query_books_by_author(kg, "乔治・奥威尔")
    for i, book in enumerate(orwell_books, 1):
        print(f"{i}. {book}")
    
    # 保存知识图谱到文件
    kg.serialize("./KG-Class/Project3/books_authors_knowledge_graph.ttl", format="turtle")
    print("\n知识图谱已保存到 './KG-Class/Project3/books_authors_knowledge_graph.ttl'")

if __name__ == "__main__":
    main()