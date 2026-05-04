print("debug")
from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
from sqlalchemy import text
import requests
import time
from urllib3.util import Retry
from requests.adapters import HTTPAdapter
REQUEST_TIMEOUT = 30   # 设置超时时间为30秒
import numpy as np
from scipy.stats import pearsonr
from werkzeug.security import generate_password_hash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask import redirect
from flask import flash, url_for
from sqlalchemy import or_
# 配置选项：是否每次启动都强制重建隐式模型（设为 True 则每次都重建，False 则加载已有模型）
FORCE_REBUILD_IMPLICIT = True   # 可根据需要修改

TMDB_API_KEY = '5ed7cbe0bb8d1a76132ccc8a453ec377'
# 两个认证方式选一个即可，推荐用 API Key

app = Flask(__name__)

# 数据库配置
# 数据库配置：优先使用环境变量（Railway），本地开发时保留默认值
DATABASE_URL = os.environ.get('DATABASE_URL', 'mysql+pymysql://root:150437@localhost/movie_db')
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL

# 为 Aiven 云端 MySQL 强制开启 SSL (无需证书)
if 'DATABASE_URL' in os.environ:
    app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
        'connect_args': {
            'ssl': {'fake_flag_to_enable_tls': True}
        }
    }

# 启动时打印数据库连接信息（密码已隐藏）
_DB_DISPLAY = DATABASE_URL.replace(DATABASE_URL.split('@')[0].split(':')[-1], '****') if '@' in DATABASE_URL else DATABASE_URL
print(f"🚀 数据库连接：{_DB_DISPLAY}")

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'sdjfksdjhfkjsdhfkj'

db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'  # 未登录时跳转到登录页面

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

session = requests.Session()
retries = Retry(
    total=3,
    connect=3,          # 连接超时重试3次
    read=3,             # 读取超时重试3次
    backoff_factor=1,
    status_forcelist=[500, 502, 503, 504]
)
session.mount('https://', HTTPAdapter(max_retries=retries))

def fetch_movie_details_from_tmdb(movie_id):
    print(f"开始处理电影 {movie_id}")
    from sqlalchemy import text
    result = db.session.execute(
        text('SELECT tmdbId FROM links WHERE movieId = :mid'),
        {'mid': movie_id}
    ).fetchone()
    if not result or not result[0]:
        print(f"电影 {movie_id} 没有对应的 tmdbId")
        return None, None, None
    
    tmdb_id = int(result[0])
    url = f'https://api.themoviedb.org/3/movie/{tmdb_id}'
    params = {
        'api_key': TMDB_API_KEY,
        'append_to_response': 'credits'
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code != 200:
                print(f"TMDB API 请求失败: {resp.status_code} (尝试 {attempt+1}/{max_retries})")
                if resp.status_code == 404:
                    db.session.execute(
                        text('UPDATE movies SET tmdb_updated = TRUE, tmdb_skipped = TRUE WHERE id = :id'),
                        {'id': movie_id}
                    )
                    db.session.commit()
                    return None, None, None
                if attempt == max_retries - 1:
                    return None, None, None
                time.sleep(2 ** attempt)
                continue
            
            data = resp.json()
            
            # 提取导演
            director = None
            if 'credits' in data and 'crew' in data['credits']:
                for person in data['credits']['crew']:
                    if person['job'] == 'Director':
                        director = person['name']
                        break
            
            # 提取演员（前5）
            actors = []
            if 'credits' in data and 'cast' in data['credits']:
                for person in data['credits']['cast'][:5]:
                    actors.append(person['name'])
            actors_str = ', '.join(actors) if actors else None
            
            # 提取海报URL
            poster_path = data.get('poster_path')
            poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None
            
            # 提取简介
            overview = data.get('overview', '')
            
            # 更新数据库（一次性更新所有字段）
            db.session.execute(
                text('''
                    UPDATE movies 
                    SET director = :director, 
                        actors = :actors, 
                        poster_url = :poster,
                        overview = :overview,
                        tmdb_updated = TRUE
                    WHERE id = :id
                '''),
                {
                    'director': director,
                    'actors': actors_str,
                    'poster': poster_url,
                    'overview': overview,
                    'id': movie_id
                }
            )
            db.session.commit()
            print(f"✅ 电影 {movie_id} 信息已更新（导演: {director}, 演员: {actors_str}, 海报: {poster_url}）")
            return director, actors_str, poster_url
        
        except Exception as e:
            print(f"请求异常: {e} (尝试 {attempt+1}/{max_retries})")
            if attempt == max_retries - 1:
                return None, None, None
            time.sleep(2 ** attempt)
    
    return None, None, None

# ---------- 模型定义必须放在 db 初始化之后 ----------
class User(db.Model, UserMixin):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
    role = db.Column(db.String(20), default='user')  # 'user' 或 'admin'

class Movie(db.Model):
    __tablename__ = 'movies'
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    genres = db.Column(db.String(100))
    rating = db.Column(db.Float, default=0)
    director = db.Column(db.String(255))   # 新增
    actors = db.Column(db.Text)            # 新增
    poster_url = db.Column(db.String(500)) # 新增
    tmdb_updated = db.Column(db.Boolean, default=False)
    # 如果你还加了 overview，也加上
    overview = db.Column(db.Text)

class Rating(db.Model):
    __tablename__ = 'ratings'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    movie_id = db.Column(db.Integer, db.ForeignKey('movies.id'))
    rating = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

class GenreLibrary(db.Model):
    __tablename__ = 'genre_library'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)

class SystemLog(db.Model):
    __tablename__ = 'system_log'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    username = db.Column(db.String(80))
    action = db.Column(db.String(100))
    target_type = db.Column(db.String(50))
    target_id = db.Column(db.Integer)
    details = db.Column(db.Text)
    ip_address = db.Column(db.String(45))
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
# -------------------------------------------------

def get_filtered_movies(title=None, genre=None, director=None, actor=None, page=1, per_page=20):
    query = Movie.query
    if title:
        query = query.filter(Movie.title.ilike(f'%{title}%'))
    if genre:
        query = query.filter(Movie.genres.like(f'%{genre}%'))   # genres 字段是 '|' 分隔
    if director:
        query = query.filter(Movie.director.ilike(f'%{director}%'))
    if actor:
        query = query.filter(Movie.actors.ilike(f'%{actor}%'))
    paginated = query.paginate(page=page, per_page=per_page, error_out=False)
    return paginated

def get_rating_counts(movie_ids):
    """
    传入电影ID列表，返回字典 {movie_id: rating_count}
    """
    if not movie_ids:
        return {}
    # 注意：IN 子句需要元组，且 SQLAlchemy 的 text 不支持直接传列表，需使用 tuple
    result = db.session.execute(
        text('SELECT movie_id, COUNT(*) FROM ratings WHERE movie_id IN :ids GROUP BY movie_id'),
        {'ids': tuple(movie_ids)}
    ).fetchall()
    return {row[0]: row[1] for row in result}

def log_action(user_id, action, target_type=None, target_id=None, details=None):
    user = User.query.get(user_id)
    username = user.username if user else 'unknown'
    ip = request.remote_addr
    log = SystemLog(
        user_id=user_id,
        username=username,
        action=action,
        target_type=target_type,
        target_id=target_id,
        details=details,
        ip_address=ip
    )
    db.session.add(log)
    db.session.commit()

@app.route('/popular')
def popular():
    page = request.args.get('page', 1, type=int)
    per_page = 10
    offset = (page - 1) * per_page
    sql = text('''
        SELECT m.id, m.title, m.genres, COUNT(r.movie_id) as rating_count, m.poster_url
        FROM movies m
        LEFT JOIN ratings r ON m.id = r.movie_id
        GROUP BY m.id
        ORDER BY rating_count DESC
        LIMIT :limit OFFSET :offset
    ''')
    result = db.session.execute(sql, {'limit': per_page, 'offset': offset}).fetchall()
    movies = [{'id': row[0], 'title': row[1], 'genres': row[2], 'rating_count': row[3], 'poster_url': row[4]} for row in result]
    has_next = len(result) == per_page
    return {'popular': movies, 'has_next': has_next}

@app.route('/')
def index():
    if current_user.is_authenticated and current_user.role == 'admin':
        return redirect('/admin')
    return render_template('index.html', logged_in=current_user.is_authenticated)

# ---------- 基于内容的推荐模型 ----------
class ContentBasedRecommender:
    def __init__(self):
        self.movies_df = None
        self.similarity_matrix = None
        self.movie_id_to_idx = None
    
    def build(self):
        if os.path.exists('content_based_model.pkl'):
            self.load()
            print("跳过内容模型重建（无.pkl文件）")
            return
        # 原来重建代码暂时不执行，防止内存爆炸
        """从数据库读取电影数据，分别对类型、导演、演员进行向量化并加权合并"""
        from sqlalchemy import text
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import pandas as pd
        import pickle
    
        # 1. 读取数据
        result = db.session.execute(
            text('SELECT id, title, genres, director, actors FROM movies')
        )
        rows = list(result)
        
        self.movies_df = pd.DataFrame([{
            'id': r.id,
            'title': r.title,
            'genres': r.genres or '',
            'director': r.director or '',
            'actors': r.actors or ''
        } for r in rows])
        
        # 2. 分别向量化各特征
        # 类型（保持原样）
        genres_text = self.movies_df['genres'].fillna('').apply(lambda x: x.replace('|', ' '))
        tfidf_genres = TfidfVectorizer()
        genres_matrix = tfidf_genres.fit_transform(genres_text)
        
        # 导演（每个导演名作为一个词）
        director_text = self.movies_df['director'].fillna('')
        tfidf_director = TfidfVectorizer()
        director_matrix = tfidf_director.fit_transform(director_text)
        
        # 演员（多个演员名，用空格分开）
        actors_text = self.movies_df['actors'].fillna('').apply(lambda x: x.replace(',', ' '))
        tfidf_actors = TfidfVectorizer()
        actors_matrix = tfidf_actors.fit_transform(actors_text)
        
        # 3. 设置权重（可以自由调整）
        weight_genres = 1.0    # 类型权重
        weight_director = 2.0  # 导演权重
        weight_actors = 1.5    # 演员权重
        
        # 4. 加权合并（按列拼接）
        from scipy.sparse import hstack
        combined_matrix = hstack([
            genres_matrix * weight_genres,
            director_matrix * weight_director,
            actors_matrix * weight_actors
        ])
        
        # 5. 计算相似度矩阵
        self.similarity_matrix = cosine_similarity(combined_matrix)
        # 建立 id -> index 映射
        self.movie_id_to_idx = pd.Series(
            self.movies_df.index, 
            index=self.movies_df['id']
        ).to_dict()
        # 6. 保存模型（包括各个向量器，以便后续对新电影推荐）
        self.vectorizers = {
            'genres': tfidf_genres,
            'director': tfidf_director,
            'actors': tfidf_actors
        }
        self.matrix = combined_matrix
        
        with open('content_based_model.pkl', 'wb') as f:
            pickle.dump({
                'movies_df': self.movies_df,
                'similarity_matrix': self.similarity_matrix,
                'vectorizers': self.vectorizers,
                'weights': (weight_genres, weight_director, weight_actors),  # ← 这里加逗号
                'movie_id_to_idx': self.movie_id_to_idx
            }, f)
        
        print(f"✅ 特征加权模型构建完成！权重：类型={weight_genres}, 导演={weight_director}, 演员={weight_actors}")
    
    def load(self):
        if os.path.exists('content_based_model.pkl'):
            with open('content_based_model.pkl', 'rb') as f:
                data = pickle.load(f)
                self.movie_id_to_idx = data.get('movie_id_to_idx')
                self.movies_df = data['movies_df']
                self.similarity_matrix = data['similarity_matrix']
                self.vectorizers = data.get('vectorizers')  # 兼容旧模型
                print("模型加载成功")
                return True
        return False
    
    def recommend(self, movie_id, top_n=10):
        """返回与 movie_id 最相似的 top_n 部电影（不包括自身）"""
        if self.similarity_matrix is None:
            return []
        idx = self.movie_id_to_idx.get(movie_id)
        if idx is None:
            return []
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        # 按相似度降序排序，排除自身
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
        similar_indices = [i[0] for i in sim_scores]
        similar_movies = self.movies_df.iloc[similar_indices][['id', 'title', 'genres']].to_dict('records')
        # 把 genres 中的空格还原为 '|'（可选）
        for m in similar_movies:
            m['genres'] = m['genres'].replace(' ', '|')
        return similar_movies

# 创建全局推荐器实例
recommender = ContentBasedRecommender()

class PearsonCF:
    def __init__(self, top_k=1000, model_path='pearson_cf.pkl'):
        self.top_k = top_k
        self.model_path = model_path
        self.movie_ids = None
        self.sim_matrix = None
        self.id_to_idx = None

    def build(self):
        if os.path.exists(self.model_path):
            # 加载已有文件
            print("跳过皮尔逊模型重建（无.pkl文件）")
            return
        # 如果已有保存的模型，直接加载
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
                self.movie_ids = data['movie_ids']
                self.sim_matrix = data['sim_matrix']
                self.id_to_idx = data['id_to_idx']
                print(f"✅ 已加载协同过滤模型（包含 {len(self.movie_ids)} 部电影）")
                return

        # 否则重新构建
        top = db.session.execute(
            text('SELECT movie_id, COUNT(*) as cnt FROM ratings GROUP BY movie_id ORDER BY cnt DESC LIMIT :k'),
            {'k': self.top_k}
        ).fetchall()
        movie_ids = [row[0] for row in top]
        df = pd.read_sql(
            f'SELECT user_id, movie_id, rating FROM ratings WHERE movie_id IN ({",".join(["%s"]*len(movie_ids))})',
            db.engine, params=tuple(movie_ids)
        )
        
        pivot = df.pivot_table(index='user_id', columns='movie_id', values='rating').fillna(0)
        movie_mat = pivot.T.values
        self.movie_ids = pivot.columns.tolist()
        self.id_to_idx = {mid: i for i, mid in enumerate(self.movie_ids)}

        n = len(self.movie_ids)
        self.sim_matrix = np.zeros((n, n))
        for i in range(n):
            vec_i = movie_mat[i]
            for j in range(i, n):
                mask = (vec_i != 0) & (movie_mat[j] != 0)
                if np.sum(mask) < 2:
                    corr = 0
                else:
                    corr, _ = pearsonr(vec_i[mask], movie_mat[j][mask])
                self.sim_matrix[i, j] = corr
                self.sim_matrix[j, i] = corr
        
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'movie_ids': self.movie_ids,
                'sim_matrix': self.sim_matrix,
                'id_to_idx': self.id_to_idx
            }, f)
        print(f"✅ 皮尔逊协同过滤构建完成并已保存，包含 {n} 部电影")

    def recommend(self, movie_id, top_n=10):
        idx = self.id_to_idx.get(movie_id)
        if idx is None:
            return []
        sim = list(enumerate(self.sim_matrix[idx]))
        sim.sort(key=lambda x: x[1], reverse=True)
        return [(self.movie_ids[i], sim_val) for i, sim_val in sim[1:top_n+1]]

class ImplicitCF:
    def __init__(self, model_path='implicit_cf.pkl'):
        self.model_path = model_path
        if FORCE_REBUILD_IMPLICIT:
            self.build()
        else:
            self.load_or_build()

    def load_or_build(self):
        """加载已有模型，若不存在则构建"""
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
                self.movie_ids = data['movie_ids']
                self.sim_matrix = data['sim_matrix']
                self.id_to_idx = data['id_to_idx']
                print(f"✅ 已加载隐式协同过滤模型（包含 {len(self.movie_ids)} 部电影）")
                return
        self.build()

    def build(self):
        if not os.path.exists(self.model_path):
            print("跳过隐式协同过滤重建（无模型文件）")
            return
        """从数据库构建新模型并保存"""
        implicit_df = pd.read_sql('SELECT user_id, movie_id, weight FROM implicit_ratings', db.engine)
        if implicit_df.empty:
            print("⚠️ 没有隐式评分数据，跳过构建")
            self.movie_ids = []
            self.sim_matrix = np.array([])
            self.id_to_idx = {}
            return

        pivot = implicit_df.pivot_table(index='user_id', columns='movie_id', values='weight').fillna(0)
        movie_mat = pivot.T.values
        self.movie_ids = pivot.columns.tolist()
        self.id_to_idx = {mid: i for i, mid in enumerate(self.movie_ids)}

        from sklearn.metrics.pairwise import cosine_similarity
        self.sim_matrix = cosine_similarity(movie_mat)

        # 保存模型
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'movie_ids': self.movie_ids,
                'sim_matrix': self.sim_matrix,
                'id_to_idx': self.id_to_idx
            }, f)
        print(f"✅ 隐式协同过滤构建完成，包含 {len(self.movie_ids)} 部电影")

    def recommend(self, movie_id, top_n=10):
        idx = self.id_to_idx.get(movie_id)
        if idx is None:
            return []
        sim = list(enumerate(self.sim_matrix[idx]))
        sim.sort(key=lambda x: x[1], reverse=True)
        return [(self.movie_ids[i], sim_val) for i, sim_val in sim[1:top_n+1]]

# 创建皮尔逊推荐器实例
pearson_cf = PearsonCF(top_k=1000)   # 可调整采样数量
# 在应用启动时加载或构建模型
@app.before_request
def setup_recommender():
    if not hasattr(app, 'recommender_initialized'):
        # 检查模型文件是否存在且非空
        if os.path.exists('content_based_model.pkl') and os.path.getsize('content_based_model.pkl') > 0:
            recommender.load()
        else:
            print("跳过内容模型加载（文件不存在或为空）")
        app.recommender_initialized = True

# 相似电影接口
@app.route('/similar/<int:movie_id>')
def similar_movies(movie_id):
    # 先尝试从数据库获取电影信息（看是否有导演数据）
    movie = db.session.execute(
        text('SELECT director, actors FROM movies WHERE id = :id'),
        {'id': movie_id}
    ).fetchone()
    
    # 如果没有导演信息，调用 TMDB 获取并更新
    if not movie or not movie.director:
        fetch_movie_details_from_tmdb(movie_id)
    
    # 获取相似电影列表（用现有的 recommender）
    movies = recommender.recommend(movie_id, top_n=10)
    
    # 为每部相似电影补充导演和演员信息（从数据库查）
    for m in movies:
        info = db.session.execute(
            text('SELECT director, actors, poster_url FROM movies WHERE id = :id'),
            {'id': m['id']}
        ).fetchone()
        if info:
            m['director'] = info[0]
            m['actors'] = info[1]
            m['poster_url'] = info[2]
    
    return {'similar': movies}

@app.route('/test_fetch/<int:movie_id>')
def test_fetch(movie_id):
    director, actors, poster = fetch_movie_details_from_tmdb(movie_id)
    return {
        'movie_id': movie_id,
        'director': director,
        'actors': actors,
        'poster_url': poster
    }

@app.route('/pearson_similar/<int:movie_id>')
def pearson_similar(movie_id):
    similar_ids = pearson_cf.recommend(movie_id, top_n=10)
    movies = []
    for mid in similar_ids:
        info = db.session.execute(
            text('SELECT id, title, director, actors, poster_url FROM movies WHERE id = :id'),
            {'id': mid}
        ).fetchone()
        if info:
            movies.append({
                'id': info[0],
                'title': info[1],
                'director': info[2],
                'actors': info[3],
                'poster_url': info[4]
            })
    return {'pearson_similar': movies}

@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # 检查用户名是否已存在
        if User.query.filter_by(username=username).first():
            error = '用户名已存在'
        else:
            hashed_pw = generate_password_hash(password)
            user = User(username=username, password_hash=hashed_pw)
            db.session.add(user)
            db.session.commit()
            flash('注册成功，请登录', 'success')
            return redirect(url_for('login'))
    return render_template('register.html', error=error)

@app.route('/login', methods=['GET', 'POST'])
def login():
    next_url = request.args.get('next') or request.form.get('next') or '/'
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            if user.role == 'admin':
                return redirect('/admin')
            return redirect(next_url)
        error = '用户名或密码错误'
    return render_template('login.html', next_url=next_url, error=error)

@app.route('/change_password', methods=['GET', 'POST'])
@login_required
def change_password():
    if request.method == 'POST':
        old_password = request.form['old_password']
        new_password = request.form['new_password']
        confirm_password = request.form['confirm_password']
        
        if not check_password_hash(current_user.password_hash, old_password):
            return '旧密码错误'
        if new_password != confirm_password:
            return '两次新密码不一致'
        
        current_user.password_hash = generate_password_hash(new_password)
        db.session.commit()
        return '密码修改成功！<a href="/">返回首页</a>'
    
    return '''
        <form method="post">
            <input type="password" name="old_password" placeholder="旧密码" required><br>
            <input type="password" name="new_password" placeholder="新密码" required><br>
            <input type="password" name="confirm_password" placeholder="确认新密码" required><br>
            <button type="submit">修改密码</button>
        </form>
    '''

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect('/')

@app.route('/rate_movie', methods=['POST'])
@login_required
def rate_movie():
    movie_id = request.form.get('movie_id')
    rating_str = request.form.get('rating')
    if not movie_id or not rating_str:
        return '参数错误', 400

    try:
        rating = float(rating_str)
    except ValueError:
        return '评分必须为数字', 400

    if rating < 0.1 or rating > 5:
        return '评分必须在0.1到5之间', 400

    existing = Rating.query.filter_by(user_id=current_user.id, movie_id=movie_id).first()
    if existing:
        existing.rating = rating
    else:
        new_rating = Rating(user_id=current_user.id, movie_id=movie_id, rating=rating)
        db.session.add(new_rating)
    db.session.commit()
    return '评分成功'

@app.route('/movie/<int:movie_id>')
def movie_detail(movie_id):
    print(f"正在查询电影 ID: {movie_id}")
    # 获取来源参数，默认 'direct'
    src = request.args.get('src', 'direct')
    
    # 定义权重映射
    weight_map = {
        'home': 0.5,      # 首页点击
        'search': 0.8,    # 搜索点击
        'recommend': 0.8, # 推荐跳转
        'direct': 0.5     # 直接访问
    }
    weight = weight_map.get(src, 0.5)
    
    # 映射来源到对应的列名
    column_map = {
        'home': 'cnt_home',
        'search': 'cnt_search',
        'recommend': 'cnt_recommend',
        'direct': 'cnt_direct'
    }
    col = column_map.get(src, 'cnt_direct')

    if current_user.is_authenticated:
        db.session.execute(
            text(f'''
                INSERT INTO implicit_ratings (user_id, movie_id, weight, {col})
                VALUES (:uid, :mid, :weight, 1)
                ON DUPLICATE KEY UPDATE 
                    weight = weight + :weight,
                    {col} = {col} + 1,
                    updated_at = NOW()
            '''),
            {'uid': current_user.id, 'mid': movie_id, 'weight': weight}
        )
        db.session.commit()
    # 查询电影基本信息
    movie = db.session.execute(
    text('SELECT id, title, genres, director, actors, poster_url, overview FROM movies WHERE id = :id'),
    {'id': movie_id}
    ).fetchone()
    print(f"查询结果: {movie}")
    if not movie:
        print("电影不存在，返回404")
        return "电影不存在", 404
    
    # 插入点击行为记录（仅当用户登录时）
    if current_user.is_authenticated:
        db.session.execute(
            text('''
                INSERT INTO user_behavior (user_id, behavior_type, target_id)
                VALUES (:uid, 'click_movie', :mid)
            '''),
            {'uid': current_user.id, 'mid': movie_id}
        )
        db.session.commit()

    # 查询该电影的平均分和评分人数
    rating_stats = db.session.execute(
        text('''
            SELECT AVG(rating) as avg_rating, COUNT(*) as rating_count
            FROM ratings
            WHERE movie_id = :movie_id
        '''),
        {'movie_id': movie_id}
    ).fetchone()
    
    avg_rating = round(rating_stats[0], 1) if rating_stats[0] else 0
    rating_count = rating_stats[1] if rating_stats[1] else 0
    
    print("渲染模板")
    # 在 movie_detail 函数内，获取电影信息后、渲染模板前添加
    user_type = 'new'  # 默认新用户
    if current_user.is_authenticated:
        rated_count = Rating.query.filter_by(user_id=current_user.id).count()
        clicked_count = db.session.execute(
            text('SELECT COUNT(*) FROM implicit_ratings WHERE user_id = :uid'),
            {'uid': current_user.id}
        ).scalar()
        if rated_count > 0:
            user_type = 'old'
        elif clicked_count > 0:
            user_type = 'implicit'

    print(f"传递给模板的 movie.overview: {movie.overview}")  # 新增

    # 在 render_template 中增加 user_type
    return render_template('movie_detail.html',
                        movie=movie,
                        avg_rating=avg_rating,
                        rating_count=rating_count,
                        user_type=user_type,
                        logged_in=current_user.is_authenticated)   # 新增

@app.route('/admin')
@login_required
def admin():
    if current_user.role != 'admin':
        return "无权访问", 403

    # 获取筛选参数
    title = request.args.get('title', '')
    genres = request.args.getlist('genre')     # 多选类型列表
    director = request.args.get('director', '')
    actors = request.args.getlist('actor')     # 多选演员列表
    # 只有执行了筛选操作（即有任何筛选参数）才显示列表
    show_list = any([title, genres, director, actors])
    page = request.args.get('page', 1, type=int)
    per_page = 20

    # 总电影数
    total_movies = Movie.query.count()

    # 构建查询
    query = Movie.query
    if title:
        query = query.filter(Movie.title.ilike(f'%{title}%'))
    if genres:
        # 对于每个选中的类型，需要检查 genres 字段包含该类型（用 `|` 分隔）
        genre_conditions = [Movie.genres.like(f'%{g}%') for g in genres]
        query = query.filter(or_(*genre_conditions))
    if director:
        query = query.filter(Movie.director.in_(director))
    if actors:
        actor_conditions = [Movie.actors.like(f'%{a}%') for a in actors]
        query = query.filter(or_(*actor_conditions))
    filtered_count = query.count()
    paginated = query.paginate(page=page, per_page=per_page, error_out=False) if show_list else None

    # 获取所有类型、导演、演员下拉列表（用于筛选表单）
    all_movies = Movie.query.all()
    all_genres = sorted(set(g for m in all_movies for g in m.genres.split('|')))
    all_directors = sorted(set(m.director for m in all_movies if m.director))
    all_actors = set()
    for m in all_movies:
        if m.actors:
            for a in m.actors.split(', '):
                all_actors.add(a)
    all_actors = sorted(all_actors)

    # 原有统计数据不变
    hot_searches = db.session.execute(
        text('SELECT search_query, COUNT(*) as cnt FROM user_behavior WHERE behavior_type = "search" AND search_query IS NOT NULL GROUP BY search_query ORDER BY cnt DESC LIMIT 10')
    ).fetchall()
    active_users = db.session.execute(
        text('SELECT user_id, COUNT(*) as cnt FROM user_behavior WHERE behavior_type = "click_movie" GROUP BY user_id ORDER BY cnt DESC LIMIT 10')
    ).fetchall()
    feedbacks = db.session.execute(
        text('SELECT id, user_id, search_query, movie_title, status, created_at FROM feedback ORDER BY created_at DESC LIMIT 50')
    ).fetchall()

    recent_logs = SystemLog.query.order_by(SystemLog.created_at.desc()).limit(20).all()

    hot_labels = [row[0] for row in hot_searches]
    hot_data = [row[1] for row in hot_searches]
    user_labels = [f'用户{row[0]}' for row in active_users]
    user_data = [row[1] for row in active_users]

    genre_rows = db.session.execute(
        text('SELECT genres FROM movies')
    ).fetchall()

    genre_counter = {}
    for row in genre_rows:
        if row[0]:  # genres 字段不为空
            for g in row[0].split('|'):
                g = g.strip()
                if g:
                    genre_counter[g] = genre_counter.get(g, 0) + 1

    # 按数量降序排列，取前15个
    genre_sorted = sorted(genre_counter.items(), key=lambda x: x[1], reverse=True)[:15]
    genre_labels = [g[0] for g in genre_sorted]
    genre_counts = [g[1] for g in genre_sorted]

    return render_template('admin.html',
                           show_list=show_list,
                           movies=paginated.items if paginated else [],
                           pagination=paginated,
                           total_movies=total_movies,
                           filtered_count=filtered_count,
                           title=title, genre=genres, director=director, actor=actors,
                           all_genres=all_genres,
                           all_directors=all_directors,
                           all_actors=all_actors,
                           hot_searches=hot_searches,
                           active_users=active_users,
                           hot_labels=hot_labels,
                           hot_data=hot_data,
                           user_labels=user_labels,
                           user_data=user_data,
                           recent_logs=recent_logs,
                           feedbacks=feedbacks,
                           genre_labels=genre_labels,
                           genre_counts=genre_counts)

@app.route('/recommend_for_me')
@login_required
def recommend_for_me():
    # 获取用户评分过的电影（≥4分）
    liked = Rating.query.filter_by(user_id=current_user.id).filter(Rating.rating >= 4).all()
    if not liked:
        # 如果没有显式评分，尝试用隐式评分
        implicit_liked = db.session.execute(
            text('SELECT movie_id FROM implicit_ratings WHERE user_id = :uid ORDER BY weight DESC LIMIT 5'),
            {'uid': current_user.id}
        ).fetchall()
        if not implicit_liked:
            return {'recommend': []}
        liked_movies = [row[0] for row in implicit_liked]
    else:
        liked_movies = [r.movie_id for r in liked]

    # 收集候选电影及其得分
    scores = {}
    weight_explicit = 0.7
    weight_implicit = 0.3

    for mid in liked_movies:
        # 从显式协同过滤获取相似电影
        for sim_mid, sim_val in pearson_cf.recommend(mid, top_n=10):
            scores[sim_mid] = scores.get(sim_mid, 0) + weight_explicit * sim_val
        # 从隐式协同过滤获取相似电影
        for sim_mid, sim_val in implicit_cf.recommend(mid, top_n=10):
            scores[sim_mid] = scores.get(sim_mid, 0) + weight_implicit * sim_val

    # 排除已看过的电影
    rated_ids = [r.movie_id for r in Rating.query.filter_by(user_id=current_user.id).all()]
    candidates = [(mid, score) for mid, score in scores.items() if mid not in rated_ids]
    candidates.sort(key=lambda x: x[1], reverse=True)
    top_ids = [mid for mid, _ in candidates[:10]]

    # 获取电影详细信息
    movies = []
    for mid in top_ids:
        info = db.session.execute(
            text('SELECT id, title, genres, director, actors, poster_url FROM movies WHERE id = :id'),
            {'id': mid}
        ).fetchone()
        if info:
            movies.append({
                'id': info[0],
                'title': info[1],
                'genres': info[2],
                'director': info[3],
                'actors': info[4],
                'poster_url': info[5],
                'rating_count': None
            })
    return {'recommend': movies}

@app.route('/search')
def search():
    keyword = request.args.get('q', '').strip()
    if not keyword:
        return redirect('/browse')  # 无关键词时直接跳转到电影库
    movies = Movie.query.filter(Movie.title.ilike(f'%{keyword}%')).all()
    return render_template('search_results.html', movies=movies, keyword=keyword,logged_in=current_user.is_authenticated)

@app.route('/browse')
def browse():
    genres = request.args.getlist('genre')
    director = request.args.get('director', '')
    actors = request.args.getlist('actor')
    page = request.args.get('page', 1, type=int)
    per_page = 20

    query = Movie.query
    if genres:
        genre_conds = [Movie.genres.like(f'%{g}%') for g in genres]
        query = query.filter(or_(*genre_conds))
    if director:
        query = query.filter(Movie.director == director)
    if actors:
        actor_conds = [Movie.actors.like(f'%{a}%') for a in actors]
        query = query.filter(or_(*actor_conds))

    paginated = query.paginate(page=page, per_page=per_page, error_out=False)

    # 获取所有下拉选项
    all_movies = Movie.query.all()
    all_genres = sorted(set(g for m in all_movies for g in m.genres.split('|')))
    all_directors = sorted(set(m.director for m in all_movies if m.director))
    all_actors = set()
    for m in all_movies:
        if m.actors:
            for a in m.actors.split(', '):
                all_actors.add(a)
    all_actors = sorted(all_actors)

    return render_template('browse.html',
                           movies=paginated.items,
                           pagination=paginated,
                           genres=genres,
                           director=director,
                           actors=actors,
                           all_genres=all_genres,
                           all_directors=all_directors,
                           all_actors=all_actors,
                           logged_in=current_user.is_authenticated)

@app.route('/feedback', methods=['POST'])
def feedback():
    search_query = request.form.get('search_query')
    movie_title = request.form.get('movie_title')
    user_id = current_user.id if current_user.is_authenticated else None
    
    db.session.execute(
        text('INSERT INTO feedback (user_id, search_query, movie_title) VALUES (:uid, :sq, :mt)'),
        {'uid': user_id, 'sq': search_query, 'mt': movie_title}
    )
    db.session.commit()
    
    return '反馈已提交，感谢您的支持！<a href="/">返回首页</a>'

@app.route('/mark_feedback_done/<int:feedback_id>')
@login_required
def mark_feedback_done(feedback_id):
    if current_user.role != 'admin':
        return "无权访问", 403
    db.session.execute(
        text('UPDATE feedback SET status = "resolved" WHERE id = :id'),
        {'id': feedback_id}
    )
    db.session.commit()
    return redirect('/admin')

@app.route('/recommend_fusion')
@login_required
def recommend_fusion():
    user_id = current_user.id

    # 分页参数
    page = request.args.get('page', 1, type=int)
    per_page = 10
    offset = (page - 1) * per_page

    # ========== 1. 获取用户行为统计 ==========
    rated_count = Rating.query.filter_by(user_id=user_id).count()
    clicked_count = db.session.execute(
        text('SELECT COUNT(*) FROM implicit_ratings WHERE user_id = :uid'),
        {'uid': user_id}
    ).scalar()

    # ========== 2. 准备热门候选电影（供后续使用） ==========
    hot_sql = text('''
        SELECT m.id, COUNT(r.movie_id) as rating_count
        FROM movies m
        JOIN ratings r ON m.id = r.movie_id
        GROUP BY m.id
        ORDER BY rating_count DESC
        LIMIT 50
    ''')
    hot_rows = db.session.execute(hot_sql).fetchall()
    hot_dict = {row[0]: row[1] for row in hot_rows}
    max_hot = max(hot_dict.values()) if hot_dict else 1  # 用于归一化

    # ========== 3. 用户分类处理 ==========

    # ----- 分支 A：纯新用户（无评分、无点击） -----
    if rated_count == 0 and clicked_count == 0:
        # 构建候选列表 (movie_id, score)，score 采用热度评分次数（无需归一化，排序相同）
        results = [(mid, hot_dict[mid]) for mid in hot_dict.keys()]
        # 按分数降序排序（热度高的在前）
        results.sort(key=lambda x: x[1], reverse=True)

        # 分页
        paginated = results[offset:offset+per_page]
        has_next = len(results) > offset + per_page

        # 获取电影详情
        movies = []
        for mid, _ in paginated:
            info = db.session.execute(
                text('SELECT id, title, genres, director, actors, poster_url FROM movies WHERE id = :id'),
                {'id': mid}
            ).fetchone()
            if info:
                movies.append({
                    'id': info[0],
                    'title': info[1],
                    'genres': info[2],
                    'director': info[3],
                    'actors': info[4],
                    'poster_url': info[5],
                    'rating_count': hot_dict[mid]
                })
        return {'recommend': movies, 'has_next': has_next}

    # ----- 分支 B：隐式用户（无评分，但有点击） -----
    elif rated_count == 0 and clicked_count > 0:
        # 获取用户点击最多的前5部电影作为种子
        seed_ids = db.session.execute(
            text('''
                SELECT movie_id FROM implicit_ratings
                WHERE user_id = :uid
                ORDER BY weight DESC
                LIMIT 5
            '''),
            {'uid': user_id}
        ).scalars().all()

        # 权重比例：热门:冷门内容:隐式协同 = 3:2:5
        w_hot = 3
        w_cold_content = 2
        w_implicit = 5
        per_page = 10

        # 1. 构建热门候选池（所有热门电影，按热度降序）
        hot_candidates = [mid for mid in hot_dict.keys()]
        hot_candidates.sort(key=lambda mid: hot_dict[mid], reverse=True)

        # 2. 构建冷门内容候选池（基于种子的内容推荐，排除热门电影）
        content_scores = {}
        for seed in seed_ids:
            recs = recommender.recommend(seed, top_n=30)  # 多取一些，保证有冷门内容
            for i, rec in enumerate(recs):
                mid = rec['id']
                sim = 1.0 - i * 0.033  # 相似度随排名递减
                content_scores[mid] = max(content_scores.get(mid, 0), sim)
        # 过滤掉热门电影
        cold_content_candidates = [mid for mid, _ in sorted(content_scores.items(), key=lambda x: x[1], reverse=True)
                                   if mid not in hot_candidates and mid not in seed_ids]

        # 3. 构建隐式协同候选池（基于种子的隐式协同推荐）
        implicit_scores = {}
        for seed in seed_ids:
            for mid, sim in implicit_cf.recommend(seed, top_n=30):
                implicit_scores[mid] = implicit_scores.get(mid, 0) + max(0, sim)
        implicit_candidates = [mid for mid, _ in sorted(implicit_scores.items(), key=lambda x: x[1], reverse=True)
                               if mid not in seed_ids]

        # 根据页码计算偏移量
        offset_hot = (page - 1) * w_hot
        offset_cold = (page - 1) * w_cold_content
        offset_implicit = (page - 1) * w_implicit

        # 取切片
        hot_slice = hot_candidates[offset_hot : offset_hot + w_hot]
        cold_slice = cold_content_candidates[offset_cold : offset_cold + w_cold_content]
        implicit_slice = implicit_candidates[offset_implicit : offset_implicit + w_implicit]

        # 合并去重（顺序：隐式优先，再冷门内容，最后热门）
        selected = []
        selected.extend(implicit_slice)
        selected.extend([mid for mid in cold_slice if mid not in selected])
        selected.extend([mid for mid in hot_slice if mid not in selected])

        # 如果不足10部，从热门池中补充（按热度顺序）
        if len(selected) < per_page:
            needed = per_page - len(selected)
            remaining_hot = [mid for mid in hot_candidates if mid not in selected]
            selected.extend(remaining_hot[:needed])

        top_ids = selected[:per_page]

        # 判断是否有下一页
        has_next = (len(hot_candidates) > offset_hot + w_hot or
                    len(cold_content_candidates) > offset_cold + w_cold_content or
                    len(implicit_candidates) > offset_implicit + w_implicit)

        # 获取电影详情
        rating_counts = get_rating_counts(top_ids)
        movies = []
        for mid in top_ids:
            info = db.session.execute(
                text('SELECT id, title, genres, director, actors, poster_url FROM movies WHERE id = :id'),
                {'id': mid}
            ).fetchone()
            if info:
                movies.append({
                    'id': info[0],
                    'title': info[1],
                    'genres': info[2],
                    'director': info[3],
                    'actors': info[4],
                    'poster_url': info[5],
                    'rating_count': rating_counts.get(mid, 0)
                })
        return {'recommend': movies, 'has_next': has_next}

    # ----- 分支 C：老用户（有评分） -----
    else:
        # 获取用户喜欢的种子（评分≥4的电影为主，不足时用点击补充）
        liked_movies = []
        explicit = Rating.query.filter_by(user_id=user_id).filter(Rating.rating >= 4).all()
        liked_movies.extend([r.movie_id for r in explicit])
        if len(liked_movies) < 5:
            implicit_rows = db.session.execute(
                text('''
                    SELECT movie_id FROM implicit_ratings
                    WHERE user_id = :uid
                    ORDER BY weight DESC
                    LIMIT :limit
                '''),
                {'uid': user_id, 'limit': 5 - len(liked_movies)}
            ).fetchall()
            liked_movies.extend([row[0] for row in implicit_rows])
        liked_movies = list(set(liked_movies))

        # 确定权重比例（每页各类应取数量）
        if rated_count < 3:
            w_hot, w_content, w_collab = 5, 3, 2
        else:
            w_hot, w_content, w_collab = 2, 3, 5

        # 构建三类候选池（去除已评分电影）
        rated_ids = [r.movie_id for r in Rating.query.filter_by(user_id=user_id).all()]

        # 1. 热门候选池（所有热门电影，按热度降序）
        hot_candidates = [mid for mid in hot_dict.keys() if mid not in rated_ids]
        hot_candidates.sort(key=lambda mid: hot_dict[mid], reverse=True)

        # 2. 内容候选池（从种子推荐中收集，按相似度降序）
        content_scores = {}
        for seed in liked_movies:
            content_recs = recommender.recommend(seed, top_n=20)
            for i, rec in enumerate(content_recs):
                mid = rec['id']
                sim = 1.0 - i * 0.1
                content_scores[mid] = max(content_scores.get(mid, 0), sim)
        content_candidates = [mid for mid, _ in sorted(content_scores.items(), key=lambda x: x[1], reverse=True) if mid not in rated_ids]

        # 3. 协同候选池（显式+隐式协同得分合并）
        collab_scores = {}
        for seed in liked_movies:
            for mid, sim in pearson_cf.recommend(seed, top_n=20):
                collab_scores[mid] = collab_scores.get(mid, 0) + max(0, sim)
            for mid, sim in implicit_cf.recommend(seed, top_n=20):
                collab_scores[mid] = collab_scores.get(mid, 0) + max(0, sim)
        collab_candidates = [mid for mid, _ in sorted(collab_scores.items(), key=lambda x: x[1], reverse=True) if mid not in rated_ids]

        # 根据页码计算各类的偏移量
        per_page = 10  # 每页数量
        offset_hot = (page - 1) * w_hot
        offset_content = (page - 1) * w_content
        offset_collab = (page - 1) * w_collab

        # 取各类的切片
        hot_slice = hot_candidates[offset_hot : offset_hot + w_hot]
        content_slice = content_candidates[offset_content : offset_content + w_content]
        collab_slice = collab_candidates[offset_collab : offset_collab + w_collab]

        # 合并并去重（保持顺序：先协同，再内容，最后热门，但比例已经确定，顺序可根据需要调整）
        selected = []
        # 先放协同（权重最高）
        selected.extend(collab_slice)
        # 再放内容
        selected.extend([mid for mid in content_slice if mid not in selected])
        # 最后放热门
        selected.extend([mid for mid in hot_slice if mid not in selected])

        # 如果某类切片的数量不足（例如候选池不够），可能会导致 total < 10
        # 此时从热门池中按顺序补充（保持不够的部分用热门填充）
        if len(selected) < per_page:
            needed = per_page - len(selected)
            # 从热门候选中取未出现在 selected 中的电影，按热度顺序
            remaining_hot = [mid for mid in hot_candidates if mid not in selected]
            selected.extend(remaining_hot[:needed])

        top_ids = selected[:per_page]

        # 判断是否有下一页（任意候选池中还有更多电影）
        has_next = (len(hot_candidates) > offset_hot + w_hot or
                    len(content_candidates) > offset_content + w_content or
                    len(collab_candidates) > offset_collab + w_collab)

        # 获取电影详情
        rating_counts = get_rating_counts(top_ids)
        movies = []
        for mid in top_ids:
            info = db.session.execute(
                text('SELECT id, title, genres, director, actors, poster_url FROM movies WHERE id = :id'),
                {'id': mid}
            ).fetchone()
            if info:
                movies.append({
                    'id': info[0],
                    'title': info[1],
                    'genres': info[2],
                    'director': info[3],
                    'actors': info[4],
                    'poster_url': info[5],
                    'rating_count': rating_counts.get(mid, 0)
                })
        return {'recommend': movies, 'has_next': has_next}

@app.route('/implicit_similar/<int:movie_id>')
def implicit_similar(movie_id):
    similar_ids = implicit_cf.recommend(movie_id, top_n=10)
    # 如果返回的是 (id, sim) 列表，提取 id
    if similar_ids and isinstance(similar_ids[0], tuple):
        similar_ids = [mid for mid, _ in similar_ids]
    movies = []
    for mid in similar_ids:
        info = db.session.execute(
            text('SELECT id, title, director, actors, poster_url FROM movies WHERE id = :id'),
            {'id': mid}
        ).fetchone()
        if info:
            movies.append({
                'id': info[0],
                'title': info[1],
                'director': info[2],
                'actors': info[3],
                'poster_url': info[4]
            })
    return {'collab': movies}  # 统一返回格式为 {'collab': [...]}

@app.route('/fused_similar/<int:movie_id>')
def fused_similar(movie_id):
    # 从显式和隐式协同各取前20部，加权融合
    pearson_list = pearson_cf.recommend(movie_id, top_n=20)  # [(id, sim), ...]
    implicit_list = implicit_cf.recommend(movie_id, top_n=20)
    
    # 权重，可根据效果调整
    w_pearson = 0.7
    w_implicit = 0.3
    
    # 合并得分
    score_dict = {}
    for mid, sim in pearson_list:
        score_dict[mid] = score_dict.get(mid, 0) + w_pearson * max(0, sim)
    for mid, sim in implicit_list:
        score_dict[mid] = score_dict.get(mid, 0) + w_implicit * max(0, sim)
    
    # 排序取前10
    sorted_items = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    movie_ids = [mid for mid, _ in sorted_items]
    
    # 获取电影详情
    movies = []
    for mid in movie_ids:
        info = db.session.execute(
            text('SELECT id, title, director, actors, poster_url FROM movies WHERE id = :id'),
            {'id': mid}
        ).fetchone()
        if info:
            movies.append({
                'id': info[0],
                'title': info[1],
                'director': info[2],
                'actors': info[3],
                'poster_url': info[4]
            })
    return {'collab': movies}

@app.route('/record_stay_time', methods=['POST'])
@login_required
def record_stay_time():
    data = request.get_json()
    movie_id = data.get('movie_id')
    duration = data.get('duration')
    if not movie_id or not duration or duration < 3:
        return '', 204

    user_id = current_user.id

    # 1. 插入 user_behavior 记录（供管理员查看）
    db.session.execute(
        text('''
            INSERT INTO user_behavior (user_id, behavior_type, target_id, duration)
            VALUES (:uid, 'stay', :mid, :dur)
        '''),
        {'uid': user_id, 'mid': movie_id, 'dur': duration}
    )

    # 2. 更新隐式评分表（影响推荐权重）
    # 停留时长权重：3-10秒 +0.1，11-30秒 +0.3，>30秒 +0.5
    if duration < 10:
        weight_inc = 0.1
    elif duration < 30:
        weight_inc = 0.3
    else:
        weight_inc = 0.5

    db.session.execute(
        text('''
            INSERT INTO implicit_ratings (user_id, movie_id, weight, stay_duration, cnt_stay)
            VALUES (:uid, :mid, :weight_inc, :dur, 1)
            ON DUPLICATE KEY UPDATE
                weight = weight + :weight_inc,
                stay_duration = stay_duration + :dur,
                cnt_stay = cnt_stay + 1,
                updated_at = NOW()
        '''),
        {'uid': user_id, 'mid': movie_id, 'weight_inc': weight_inc, 'dur': duration}
    )

    db.session.commit()
    return '', 204

# 电影管理列表
# @app.route('/admin/movies')
# @login_required
# def admin_movies():
#     if current_user.role != 'admin':
#         return "无权访问", 403
#     title = request.args.get('title', '')
#     genre = request.args.get('genre', '')
#     director = request.args.get('director', '')
#     actor = request.args.get('actor', '')
#     page = request.args.get('page', 1, type=int)

#     paginated = get_filtered_movies(title, genre, director, actor, page)
#     # 获取所有类型列表（用于下拉菜单）
#     all_genres = sorted(set(g for movie in Movie.query.all() for g in movie.genres.split('|')))
#     # 获取所有导演列表（去重）
#     all_directors = sorted(set(m.director for m in Movie.query.all() if m.director))
#     # 获取所有演员列表（需要拆分 actors 字段，比较耗时，可缓存或异步）
#     all_actors = set()
#     for movie in Movie.query.all():
#         if movie.actors:
#             for actor_name in movie.actors.split(', '):
#                 all_actors.add(actor_name)
#     all_actors = sorted(all_actors)

#     return render_template('admin_movies.html', 
#                            movies=paginated.items, 
#                            pagination=paginated,
#                            title=title, genre=genre, director=director, actor=actor,
#                            all_genres=all_genres, all_directors=all_directors, all_actors=all_actors)

# 添加电影
@app.route('/admin/movie/add', methods=['GET', 'POST'])
@login_required
def admin_movie_add():
    if current_user.role != 'admin':
        return "无权访问", 403
    # 获取所有类型列表（用于模板）
    all_genres = [g.name for g in GenreLibrary.query.order_by(GenreLibrary.name).all()]
    if request.method == 'POST':
        title = request.form['title']
        # 获取选中的类型名称列表（多选）
        selected_genres = request.form.getlist('genres')
        # 验证所有选中的类型是否都在类型库中存在
        valid_genres = [g for g in selected_genres if GenreLibrary.query.filter_by(name=g).first()]
        genres = '|'.join(valid_genres)
        director = request.form.get('director')
        actors = request.form.get('actors')
        poster_url = request.form.get('poster_url')
        overview = request.form.get('overview')
        new_movie = Movie(
            title=title,
            genres=genres,
            director=director,
            actors=actors,
            poster_url=poster_url,
            overview=overview
        )
        db.session.add(new_movie)
        db.session.commit()
        flash('电影添加成功', 'success')
        return redirect('/admin')
    return render_template('admin_movie_form.html', movie=None, all_genres=all_genres)

# 编辑电影
@app.route('/admin/movie/edit/<int:movie_id>', methods=['GET', 'POST'])
@login_required
def admin_movie_edit(movie_id):
    if current_user.role != 'admin':
        return "无权访问", 403
    movie = Movie.query.get_or_404(movie_id)
    all_genres = [g.name for g in GenreLibrary.query.order_by(GenreLibrary.name).all()]
    if request.method == 'POST':
        # 获取选中的类型名称列表
        selected_genres = request.form.getlist('genres')
        # 验证所有选中的类型是否都在类型库中存在（防止绕过前端）
        valid_genres = [g for g in selected_genres if GenreLibrary.query.filter_by(name=g).first()]
        movie.genres = '|'.join(valid_genres)
        # 处理提交
        movie.title = request.form['title']
        genres_list = request.form.getlist('genres')
        movie.genres = '|'.join(genres_list)
        movie.director = request.form.get('director')
        movie.actors = request.form.get('actors')
        movie.poster_url = request.form.get('poster_url')
        movie.overview = request.form.get('overview')
        db.session.commit()
        next_url = request.args.get('next') or url_for('admin')
        return redirect(next_url)
    return render_template('admin_movie_form.html', movie=movie, all_genres=all_genres)

# 删除电影
@app.route('/admin/movie/delete/<int:movie_id>')
@login_required
def admin_movie_delete(movie_id):
    if current_user.role != 'admin':
        return "无权访问", 403
    movie = Movie.query.get_or_404(movie_id)
    # 注意：需要先删除关联的评分记录（如果有外键约束，可设置 ON DELETE CASCADE 或手动删除）
    # 为了简单，暂时先删除评分
    Rating.query.filter_by(movie_id=movie_id).delete()
    db.session.delete(movie)
    db.session.commit()
    flash('电影已删除', 'success')
    return redirect('/admin')

@app.route('/admin/logs')
@login_required
def admin_logs():
    if current_user.role != 'admin':
        return "无权访问", 403
    page = request.args.get('page', 1, type=int)
    per_page = 20
    paginated = SystemLog.query.order_by(SystemLog.created_at.desc()).paginate(page=page, per_page=per_page, error_out=False)
    return render_template('admin_logs.html', logs=paginated.items, pagination=paginated)

if __name__ == '__main__':
    with app.app_context():
        # 协同过滤模型
        pearson_cf.build()
        # 隐式协同过滤模型（每次启动重建）
        implicit_cf = ImplicitCF()   # 这里会自动调用 build()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)