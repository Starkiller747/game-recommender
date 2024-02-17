<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>

<!-- PROJECT SHIELDS -->
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/Starkiller747/game-recommender">
  </a>

<h3 align="center">Game recommender</h3>

  <p align="center">
    recommends a game based on a given game title
    <br />
    <br />
    <br />
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about">About</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

## About
This project uses the cosine similarity to calculate which other datapoints are closest to it. The function at the end returns the top 5 games.
The datapoints were first transformed (dropping duplicates, transforming strings and scaling the values) and then vectorized to handle both numerical and categorical values.
String concatenation was used to handle the string variables.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

The following piece of code can be used to recommend games:
```
game_to_recommend = "Team Fortress 2"
recommended_games = recommend(game_to_recommend)

print(f"Recommended games for {game_to_recommend}:")
for game in recommended_games:
    print(game)
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## References

Dataset obtained from: [this dataset](https://www.kaggle.com/datasets/nikdavis/steam-store-games)

[Creative commons license 4.0](https://creativecommons.org/licenses/by/4.0/)
<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Javier Uribe - [LinkedIn](https://www.linkedin.com/in/jrus93/)

Project Link: [https://github.com/Starkiller747/game_recommender](https://github.com/Starkiller747/game_recommender)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/jrus93
