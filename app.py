try:
    from flask import Flask, render_template, redirect, url_for, flash, request, jsonify  # type: ignore[import]
    from flask_sqlalchemy import SQLAlchemy  # type: ignore[import]
    from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user  # type: ignore[import]
    from werkzeug.security import generate_password_hash, check_password_hash  # type: ignore[import]
except ImportError as exc:
    raise ImportError(
        "Missing dependencies. Install required packages: "
        "'pip install flask flask_sqlalchemy flask_login werkzeug'."
    ) from exc

from datetime import datetime, timedelta
import os

# Import our ML priority predictor
from ml_priority import PriorityPredictor

# Initialize Flask app
app = Flask(__name__)
# Secret key for session security (change in production)
app.config['SECRET_KEY'] = 'dev-secret-key-change-in-production'
# SQLite database file location
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///tasks.db'
# Disable modification tracking to save resources
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)           # Database ORM
login_manager = LoginManager(app)  # Session management
login_manager.login_view = 'login' # Where to redirect if not logged in
login_manager.login_message_category = 'info'

# Initialize ML model (global so it persists between requests)
ml_model = PriorityPredictor()

# ==================== DATABASE MODELS ====================

class User(UserMixin, db.Model):
    """User model for authentication. UserMixin adds required methods for Flask-Login."""
    id = db.Column(db.Integer, primary_key=True)           # Unique ID
    username = db.Column(db.String(80), unique=True, nullable=False)  # Login name
    email = db.Column(db.String(120), unique=True, nullable=False)    # Email address
    password_hash = db.Column(db.String(200), nullable=False)         # Encrypted password
    
    # Relationship: one user has many tasks
    tasks = db.relationship('Task', backref='owner', lazy=True, cascade='all, delete-orphan')

    def set_password(self, password):
        """Hash and store password securely."""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Verify password against stored hash."""
        return check_password_hash(self.password_hash, password)


class Task(db.Model):
    """Task model with ML-predicted priority."""
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)        # Task name
    description = db.Column(db.Text, nullable=False)         # Task details (used by ML)
    priority = db.Column(db.Integer, nullable=False, default=2)  # 1=Low, 2=Medium, 3=High
    deadline = db.Column(db.DateTime, nullable=False)          # Due date
    status = db.Column(db.String(20), default='pending')     # pending or completed
    created_at = db.Column(db.DateTime, default=datetime.utcnow)  # Auto-set timestamp
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)  # Owner

    def priority_label(self):
        """Return human-readable priority string."""
        labels = {1: 'Low', 2: 'Medium', 3: 'High'}
        return labels.get(self.priority, 'Medium')
    
    def is_overdue(self):
        """Check if task deadline has passed."""
        return datetime.utcnow() > self.deadline and self.status == 'pending'
    
    def days_until_deadline(self):
        """Calculate days remaining until deadline."""
        delta = self.deadline - datetime.utcnow()
        return delta.days


# ==================== FLASK-LOGIN SETUP ====================

@login_manager.user_loader
def load_user(user_id):
    """Tell Flask-Login how to load a user by ID."""
    return User.query.get(int(user_id))


# ==================== ROUTES ====================

@app.route('/')
def index():
    """Home page - redirect to dashboard if logged in, else to login."""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration page."""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        # Get form data
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm = request.form.get('confirm_password', '')
        
        # Validation
        if not username or not email or not password:
            flash('All fields are required.', 'danger')
            return redirect(url_for('register'))
        
        if password != confirm:
            flash('Passwords do not match.', 'danger')
            return redirect(url_for('register'))
        
        if len(password) < 6:
            flash('Password must be at least 6 characters.', 'danger')
            return redirect(url_for('register'))
        
        # Check if user already exists
        if User.query.filter_by(username=username).first():
            flash('Username already taken.', 'danger')
            return redirect(url_for('register'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered.', 'danger')
            return redirect(url_for('register'))
        
        # Create new user
        new_user = User(username=username, email=email)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        
        flash('Account created! Please log in.', 'success')
        return redirect(url_for('login'))
    
    # GET request - show registration form
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login page."""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        
        # Find user by username
        user = User.query.filter_by(username=username).first()
        
        # Verify credentials
        if user and user.check_password(password):
            login_user(user, remember=True)  # Create session
            flash(f'Welcome back, {username}!', 'success')
            next_page = request.args.get('next')  # Where user originally wanted to go
            return redirect(next_page) if next_page else redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password.', 'danger')
    
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    """Log out current user."""
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


@app.route('/dashboard')
@login_required
def dashboard():
    """Main dashboard showing all user tasks."""
    # Get filter from query string (default: show all)
    status_filter = request.args.get('status', 'all')
    
    # Base query: only current user's tasks
    query = Task.query.filter_by(user_id=current_user.id)
    
    # Apply status filter if not 'all'
    if status_filter == 'pending':
        query = query.filter_by(status='pending')
    elif status_filter == 'completed':
        query = query.filter_by(status='completed')
    
    # Order by: High priority first, then closest deadline
    tasks = query.order_by(Task.priority.desc(), Task.deadline.asc()).all()
    
    # Count statistics for dashboard cards
    total_tasks = Task.query.filter_by(user_id=current_user.id).count()
    pending_tasks = Task.query.filter_by(user_id=current_user.id, status='pending').count()
    completed_tasks = Task.query.filter_by(user_id=current_user.id, status='completed').count()
    high_priority = Task.query.filter_by(user_id=current_user.id, status='pending', priority=3).count()
    
    return render_template('dashboard.html', 
                         tasks=tasks, 
                         total=total_tasks,
                         pending=pending_tasks,
                         completed=completed_tasks,
                         high_priority=high_priority,
                         filter=status_filter)


@app.route('/task/add', methods=['GET', 'POST'])
@login_required
def add_task():
    """Add new task with ML auto-prioritization."""
    if request.method == 'POST':
        title = request.form.get('title', '').strip()
        description = request.form.get('description', '').strip()
        deadline_str = request.form.get('deadline', '')
        manual_priority = request.form.get('manual_priority', '')  # Optional override
        
        # Validation
        if not title or not description or not deadline_str:
            flash('Title, description, and deadline are required.', 'danger')
            return redirect(url_for('add_task'))
        
        # Parse deadline string to datetime object
        try:
            deadline = datetime.strptime(deadline_str, '%Y-%m-%dT%H:%M')
        except ValueError:
            flash('Invalid date format.', 'danger')
            return redirect(url_for('add_task'))
        
        # ========== ML AUTO-PRIORITIZATION ==========
        # Use ML model to predict priority based on description + deadline
        predicted_priority, confidence = ml_model.predict(description, deadline)
        
        # If user manually selected priority, use that. Otherwise use ML prediction.
        if manual_priority and manual_priority in ['1', '2', '3']:
            final_priority = int(manual_priority)
            ml_used = False
        else:
            final_priority = predicted_priority
            ml_used = True
        
        # Create task
        new_task = Task(
            title=title,
            description=description,
            priority=final_priority,
            deadline=deadline,
            user_id=current_user.id
        )
        
        db.session.add(new_task)
        db.session.commit()
        
        # Feedback message
        if ml_used:
            flash(f'Task added! ML predicted priority: {new_task.priority_label()} (confidence: {confidence}%)', 'success')
        else:
            flash(f'Task added with manual priority: {new_task.priority_label()}', 'success')
        
        return redirect(url_for('dashboard'))
    
    # GET request - show form
    return render_template('add_task.html')


@app.route('/task/edit/<int:task_id>', methods=['GET', 'POST'])
@login_required
def edit_task(task_id):
    """Edit existing task."""
    # Ensure user owns this task (security check)
    task = Task.query.filter_by(id=task_id, user_id=current_user.id).first_or_404()
    
    if request.method == 'POST':
        task.title = request.form.get('title', '').strip()
        task.description = request.form.get('description', '').strip()
        deadline_str = request.form.get('deadline', '')
        priority = request.form.get('priority', '')
        
        if not task.title or not task.description or not deadline_str:
            flash('All fields are required.', 'danger')
            return redirect(url_for('edit_task', task_id=task_id))
        
        try:
            task.deadline = datetime.strptime(deadline_str, '%Y-%m-%dT%H:%M')
        except ValueError:
            flash('Invalid date format.', 'danger')
            return redirect(url_for('edit_task', task_id=task_id))
        
        if priority in ['1', '2', '3']:
            task.priority = int(priority)
        
        db.session.commit()
        flash('Task updated successfully.', 'success')
        return redirect(url_for('dashboard'))
    
    # Format deadline for HTML datetime-local input
    deadline_str = task.deadline.strftime('%Y-%m-%dT%H:%M')
    return render_template('edit_task.html', task=task, deadline_str=deadline_str)


@app.route('/task/delete/<int:task_id>', methods=['POST'])
@login_required
def delete_task(task_id):
    """Delete a task."""
    task = Task.query.filter_by(id=task_id, user_id=current_user.id).first_or_404()
    db.session.delete(task)
    db.session.commit()
    flash('Task deleted.', 'info')
    return redirect(url_for('dashboard'))


@app.route('/task/complete/<int:task_id>', methods=['POST'])
@login_required
def complete_task(task_id):
    """Mark task as completed."""
    task = Task.query.filter_by(id=task_id, user_id=current_user.id).first_or_404()
    task.status = 'completed'
    db.session.commit()
    flash('Task marked as completed! Great job.', 'success')
    return redirect(url_for('dashboard'))


@app.route('/api/preview-priority', methods=['POST'])
@login_required
def preview_priority():
    """AJAX endpoint: Preview ML priority before submitting form."""
    data = request.get_json()
    description = data.get('description', '')
    deadline_str = data.get('deadline', '')
    
    if not description or not deadline_str:
        return jsonify({'error': 'Missing data'}), 400
    
    try:
        deadline = datetime.strptime(deadline_str, '%Y-%m-%dT%H:%M')
    except ValueError:
        return jsonify({'error': 'Invalid date'}), 400
    
    priority, confidence = ml_model.predict(description, deadline)
    labels = {1: 'Low', 2: 'Medium', 3: 'High'}
    
    return jsonify({
        'priority': priority,
        'priority_label': labels[priority],
        'confidence': confidence,
        'message': f'ML suggests {labels[priority]} priority ({confidence}% confidence)'
    })


# ==================== MAIN ENTRY POINT ====================

if __name__ == '__main__':
    # Create database tables if they don't exist
    with app.app_context():
        db.create_all()
        print("Database initialized!")
    
    # Run in debug mode (auto-reload on code changes)
    app.run(debug=True, port=5000)