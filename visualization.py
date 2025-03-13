import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_sentiment_over_time(dream_log):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=dream_log, x="date", y="sentiment", marker="o")
    plt.title("Dream Sentiment Over Time")
    plt.xlabel("Date")
    plt.ylabel("Sentiment (Compound Score)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def generate_wordcloud(theme_list):
    all_words = " ".join(theme_list)
    wordcloud = WordCloud(width=800, height=400, 
                         background_color='white',
                         colormap='viridis',
                         max_words=100,
                         contour_width=1,
                         contour_color='steelblue').generate(all_words)
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    return fig

def plot_emotion_distribution(dream_log):
    if 'emotions' not in dream_log.columns:
        return None
    
    all_emotions = []
    for emotions_str in dream_log['emotions']:
        if isinstance(emotions_str, str):
            emotions = [e.strip() for e in emotions_str.split(',')]
            all_emotions.extend(emotions)
    
    emotion_counts = pd.Series(all_emotions).value_counts()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    emotion_counts.plot(kind='bar', colormap='viridis')
    plt.title('Emotion Distribution in Dreams')
    plt.xlabel('Emotion')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_theme_correlation(dream_log):
    if len(dream_log) < 5 or 'themes' not in dream_log.columns:
        return None
    
    all_themes = set()
    for themes_str in dream_log['themes']:
        if isinstance(themes_str, str):
            themes = [t.strip() for t in themes_str.split(',')]
            all_themes.update(themes)
    
    theme_counts = {}
    for theme in all_themes:
        count = sum(1 for themes_str in dream_log['themes'] 
                  if isinstance(themes_str, str) and theme in themes_str)
        theme_counts[theme] = count
    
    top_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    top_theme_names = [t[0] for t in top_themes]
    
    theme_matrix = pd.DataFrame(index=dream_log.index, columns=top_theme_names)
    for theme in top_theme_names:
        theme_matrix[theme] = dream_log['themes'].apply(
            lambda x: 1 if isinstance(x, str) and theme in x else 0
        )
    
    corr_matrix = theme_matrix.corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
    plt.title('Theme Correlation Matrix')
    plt.tight_layout()
    return fig

def plot_interactive_sentiment_timeline(dream_log):
    if len(dream_log) < 2:
        return None
    
    dream_log = dream_log.copy()
    dream_log['date'] = pd.to_datetime(dream_log['date'])
    dream_log = dream_log.sort_values('date')
    
    dream_log['sentiment_rolling'] = dream_log['sentiment'].rolling(window=3, min_periods=1).mean()
    
    fig = px.line(dream_log, x='date', y=['sentiment', 'sentiment_rolling'], 
                 title='Dream Sentiment Timeline',
                 labels={'value': 'Sentiment Score', 'date': 'Date', 'variable': 'Metric'},
                 color_discrete_map={'sentiment': 'royalblue', 'sentiment_rolling': 'firebrick'})
    
    fig.update_layout(
        hovermode='x unified',
        legend=dict(title='', orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        xaxis_title='Date',
        yaxis_title='Sentiment Score')
    
    return fig

def plot_dream_symbol_network(dream_symbols, min_occurrences=2):
    if not dream_symbols or len(dream_symbols) < 3:
        return None
    
    symbol_counts = {symbol: count for symbol, count in dream_symbols.items() 
                    if count >= min_occurrences}
    
    if len(symbol_counts) < 3:
        return None
    
    nodes = list(symbol_counts.keys())
    node_sizes = [symbol_counts[node] * 10 for node in nodes]
    
    angles = np.linspace(0, 2*np.pi, len(nodes), endpoint=False)
    x_pos = np.cos(angles)
    y_pos = np.sin(angles)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x_pos, y=y_pos,
        mode='markers+text',
        marker=dict(size=node_sizes, color='skyblue', line=dict(width=1, color='darkblue')),
        text=nodes,
        textposition='top center',
        hoverinfo='text',
        name='Dream Symbols'
    ))
    
    fig.update_layout(
        title='Dream Symbol Network',
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white'
    )
    
    return fig

def create_dream_dashboard(dream_log):
    if len(dream_log) < 3:
        return None
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Sentiment Over Time', 
            'Emotion Distribution',
            'Dream Categories', 
            'Theme Frequency'
        ),
        specs=[
            [{'type': 'scatter'}, {'type': 'bar'}],
            [{'type': 'pie'}, {'type': 'bar'}]
        ]
    )
    
    dream_log = dream_log.copy()
    dream_log['date'] = pd.to_datetime(dream_log['date'])
    dream_log = dream_log.sort_values('date')
    
    fig.add_trace(
        go.Scatter(x=dream_log['date'], y=dream_log['sentiment'], mode='lines+markers',
                 name='Sentiment', line=dict(color='royalblue')),
        row=1, col=1
    )
    
    if 'emotions' in dream_log.columns:
        all_emotions = []
        for emotions_str in dream_log['emotions']:
            if isinstance(emotions_str, str):
                emotions = [e.strip() for e in emotions_str.split(',')]
                all_emotions.extend(emotions)
        
        emotion_counts = pd.Series(all_emotions).value_counts().head(5)
        
        fig.add_trace(
            go.Bar(x=emotion_counts.index, y=emotion_counts.values, name='Emotions',
                  marker_color='mediumseagreen'),
            row=1, col=2
        )
    
    if 'category' in dream_log.columns:
        category_counts = dream_log['category'].value_counts()
        fig.add_trace(
            go.Pie(labels=category_counts.index, values=category_counts.values,
                   name='Categories'),
            row=2, col=1
        )
    
    if 'themes' in dream_log.columns:
        all_themes = []
        for themes_str in dream_log['themes']:
            if isinstance(themes_str, str):
                themes = [t.strip() for t in themes_str.split(',')]
                all_themes.extend(themes)
        
        theme_counts = pd.Series(all_themes).value_counts().head(10)
        
        fig.add_trace(
            go.Bar(x=theme_counts.index, y=theme_counts.values, name='Themes',
                  marker_color='coral'),
            row=2, col=2
        )
    
    fig.update_layout(
        height=800,
        showlegend=False,
        title_text="Dream Analysis Dashboard"
    )
    
    return fig