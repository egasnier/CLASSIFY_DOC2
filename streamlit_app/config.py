"""

Config file for Streamlit App

"""

from member import Member


TITLE = "PyDOCR App"

TEAM_MEMBERS = [
    Member(
        name="Bogdan POSEA",
        linkedin_url="https://www.linkedin.com/in/bogdan-posea-a9324b38/",
        github_url="https://github.com/poseab",
    ),
    Member(
        name="Eric GASNIER",
        linkedin_url="https://www.linkedin.com/in/ericgasnier",
    ),
    Member("Jane Doe"),
]

PROMOTION = "Promotion Data Scientist - April 2022"
