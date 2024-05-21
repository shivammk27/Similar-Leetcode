import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from flask import Flask, request, render_template_string

# Load the dataframe from the pickle file
with open('embeddings.pkl', 'rb') as file:
    df = pickle.load(file)

# Load the CSV file containing problem tags and difficulty levels
csv_df = pd.read_csv('leetcode_q_full_info.csv')

# Ensure the column names are consistent
csv_df = csv_df.rename(columns={'title_slug': 'file name'})

# Merge the embeddings dataframe with the CSV dataframe
merged_df = pd.merge(df, csv_df, on='file name')

# Function to get similar problems based on the problem name


def get_similar_problems(problem_name, df, top_n=20):
    try:
        # Find the embedding for the given problem name
        embedding = df[df['file name'] == problem_name]['embedding'].values[0]
        # Compute cosine similarities between the given embedding and all other embeddings
        similarities = cosine_similarity([embedding], df['embedding'].tolist())
        # Get the top_n most similar problems
        similar_indices = similarities[0].argsort()[-top_n-1:-1][::-1]
        similar_problems = df.iloc[similar_indices].copy()
        # Add a column for the problem links
        similar_problems['Link'] = 'https://leetcode.com/problems/' + \
            similar_problems['file name'] + '/'
        return similar_problems
    except IndexError:
        return None


app = Flask(__name__)

# HTML template with CSS
template = """
<!DOCTYPE html>
<html>
<head>
    <title>LeetCode Problem Finder</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
        }
        h1 {
            color: #333;
        }
        form {
            margin-bottom: 20px;
        }
        label, input {
            font-size: 1.2em;
        }
        input[type="text"] {
            width: 60%;
            padding: 10px;
            margin: 10px 0;
            box-sizing: border-box;
        }
        input[type="submit"] {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 1em;
        }
        input[type="submit"]:hover {
            background-color: #45a049;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        table, th, td {
            border: 1px solid #ddd;
        }
        th, td {
            padding: 12px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        a {
            color: #1a0dab;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <h1>LeetCode Problem Finder</h1>
    <form method="post">
        <label for="url">Enter a LeetCode problem URL to find similar problems:</label><br>
        <input type="text" id="url" name="url" size="50" value="{{ url }}"><br><br>
        <input type="submit" value="Find Similar Problems">
    </form>
    {% if similar_problems %}
        <h2>Similar Problems:</h2>
        <table>
            <tr>
                <th>Problem Name</th>
                <th>Link</th>
                <th>Tags</th>
                <th>Difficulty</th>
            </tr>
            {% for problem in similar_problems %}
                <tr>
                    <td>{{ problem['file name'] }}</td>
                    <td><a href="{{ problem['Link'] }}" target="_blank">Link</a></td>
                    <td>{{ problem['tags'] }}</td>
                    <td>{{ problem['difficulty'] }}</td>
                </tr>
            {% endfor %}
        </table>
    {% elif error %}
        <p style="color: red;">{{ error }}</p>
    {% endif %}
</body>
</html>
"""


@app.route('/', methods=['GET', 'POST'])
def index():
    similar_problems = None
    error = None
    url = ""
    if request.method == 'POST':
        url = request.form['url']
        try:
            problem_name = url.split('/problems/')[1].strip('/')
            similar_problems_df = get_similar_problems(problem_name, merged_df)
            if similar_problems_df is not None:
                similar_problems = similar_problems_df.to_dict(
                    orient='records')
            else:
                error = "Problem not found. Please check the URL and try again."
        except Exception as e:
            error = "Invalid URL format. Please enter a valid LeetCode problem URL."
    return render_template_string(template, similar_problems=similar_problems, error=error, url=url)


if __name__ == '__main__':
    app.run(debug=True)
