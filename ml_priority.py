# ml_priority.py - Machine Learning Auto-Prioritization Engine
# -------------------------------------------------------------
# Uses Naive Bayes text classification on task descriptions
# Combined with deadline urgency scoring for final priority

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from datetime import datetime, timedelta
import numpy as np


class PriorityPredictor:
    """
    ML model that predicts task priority (1=Low, 2=Medium, 3=High)
    based on task description text and deadline proximity.
    """
    
    def __init__(self):
        """Initialize the ML pipeline."""
        # Create a text classification pipeline:
        # 1. Convert text to word count vectors (Bag of Words)
        # 2. Train Naive Bayes classifier
        self.pipeline = Pipeline([
            ('vectorizer', CountVectorizer(
                lowercase=True,      # Convert to lowercase
                stop_words='english', # Remove common words (the, and, etc.)
                max_features=100     # Limit vocabulary size
            )),
            ('classifier', MultinomialNB())
        ])
        
        # Train on synthetic dataset (cold start)
        self._train_initial_model()
    
    def _train_initial_model(self):
        """
        Create synthetic training data based on keyword rules.
        This allows the model to work immediately without user history.
        """
        # Training texts (task descriptions)
        texts = []
        labels = []
        
        # HIGH priority keywords (label = 3)
        high_keywords = [
            "urgent deadline critical emergency asap immediately important client meeting",
            "fix bug production down server crash error critical issue",
            "submit report today boss waiting review urgent approval needed",
            "call client immediately contract signing deal closing urgent",
            "security breach vulnerability fix now critical system hack",
            "payment overdue invoice urgent billing issue immediately",
            "deploy to production today release urgent hotfix needed",
            "medical emergency health critical urgent appointment",
            "fire alarm safety hazard immediate action required",
            "ceo waiting board meeting presentation urgent prepare now"
        ]
        
        # MEDIUM priority keywords (label = 2)
        medium_keywords = [
            "schedule meeting plan review next week team discussion",
            "update documentation write report prepare slides presentation",
            "code review pull request testing feature development",
            "research competitors market analysis survey feedback",
            "update website content blog post social media marketing",
            "organize files clean database optimize performance",
            "team lunch birthday celebration office event planning",
            "monthly report quarterly review performance check",
            "learn new framework tutorial course skill building",
            "backup data maintenance routine check system health"
        ]
        
        # LOW priority keywords (label = 1)
        low_keywords = [
            "someday maybe later optional nice to have low priority",
            "read article when free casual interest explore idea",
            "watch video tutorial leisure learning no rush",
            "organize desk clean office optional cleanup",
            "future project idea brainstorm someday maybe",
            "low importance minimal impact not urgent whenever",
            "personal hobby side project fun exploration",
            "wishlist consider eventually low priority task",
            "archive old files cleanup when convenient",
            "optional training voluntary no deadline flexible"
        ]
        
        # Add all examples to training set
        for text in high_keywords:
            texts.append(text)
            labels.append(3)  # High
        
        for text in medium_keywords:
            texts.append(text)
            labels.append(2)  # Medium
        
        for text in low_keywords:
            texts.append(text)
            labels.append(1)  # Low
        
        # Train the model
        self.pipeline.fit(texts, labels)
        self.is_trained = True
        
        # Store training data for future retraining
        self.training_texts = texts
        self.training_labels = labels
    
    def predict(self, description, deadline):
        """
        Predict priority for a new task.
        
        Args:
            description: Task description text (string)
            deadline: datetime object for due date
        
        Returns:
            (priority_int, confidence_percentage)
        """
        if not self.is_trained:
            return 2, 50  # Default to Medium if model not ready
        
        # Step 1: Get base priority from text classification
        # reshape for single sample prediction
        base_priority = self.pipeline.predict([description])[0]
        
        # Get prediction probabilities [prob_low, prob_medium, prob_high]
        probabilities = self.pipeline.predict_proba([description])[0]
        confidence = int(max(probabilities) * 100)  # Highest probability as confidence %
        
        # Step 2: Calculate deadline urgency boost
        now = datetime.utcnow()
        hours_until = (deadline - now).total_seconds() / 3600
        
        # Urgency rules:
        # Less than 24 hours: boost by +2 (max 3)
        # Less than 72 hours (3 days): boost by +1
        # More than 7 days: reduce by -1 (min 1)
        if hours_until < 0:
            # Already overdue - maximum priority
            urgency_boost = 2
        elif hours_until <= 24:
            urgency_boost = 2
        elif hours_until <= 72:
            urgency_boost = 1
        elif hours_until >= 168:  # 7 days
            urgency_boost = -1
        else:
            urgency_boost = 0
        
        # Step 3: Combine text prediction + deadline urgency
        # Formula: weighted average, then round
        # Text prediction counts for 60%, urgency counts for 40%
        final_score = (base_priority * 0.6) + (urgency_boost * 0.4) + 1
        
        # Ensure within bounds 1-3, then round
        final_priority = int(round(max(1, min(3, final_score))))
        
        # Adjust confidence based on how much we overrode the ML
        if final_priority != base_priority:
            confidence = max(60, confidence - 10)  # Slightly lower confidence if deadline forced change
        
        return final_priority, confidence
    
    def retrain(self, texts, labels):
        """
        Retrain model with real user data for better accuracy.
        Call this when user has 10+ manually-set priorities.
        """
        if len(texts) < 5:
            return False  # Need more data
        
        # Combine with initial training data so we don't forget basics
        combined_texts = self.training_texts + texts
        combined_labels = self.training_labels + labels
        
        self.pipeline.fit(combined_texts, combined_labels)
        return True