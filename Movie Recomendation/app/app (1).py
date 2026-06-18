
from flask import Flask, request, jsonify
import pickle, pandas as pd, numpy as np

app = Flask(__name__)

cosine_sim   = pickle.load(open("cosine_sim.pkl", "rb"))
movies_df    = pickle.load(open("movies_df.pkl", "rb"))
title_to_idx = pd.Series(movies_df.index, index=movies_df["title"]).drop_duplicates()


@app.route("/recommend", methods=["GET"])
def recommend():
    title = request.args.get("title", "")
    n     = int(request.args.get("n", 10))

    if not title:
        return jsonify({"error": "Provide a title via ?title="}), 400

    if title not in title_to_idx:
        matches = movies_df[movies_df["title"].str.contains(title, case=False, na=False)]
        if len(matches) == 0:
            return jsonify({"error": f"Movie not found: {title}"}), 404
        title = matches.iloc[0]["title"]

    idx        = title_to_idx[title]
    sim_scores = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)[1:n+1]
    indices    = [i[0] for i in sim_scores]

    result           = movies_df.iloc[indices][["title", "vote_average", "genres_parsed", "director"]].copy()
    result["similarity"] = [round(s[1], 4) for s in sim_scores]
    result["genres"] = result["genres_parsed"].apply(lambda x: ", ".join(x[:3]))
    result           = result.drop(columns=["genres_parsed"])

    return jsonify({
        "query"           : title,
        "recommendations" : result.to_dict(orient="records")
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "movies_loaded": len(movies_df)})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
