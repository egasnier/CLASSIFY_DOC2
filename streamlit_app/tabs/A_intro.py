import streamlit as st


title = "Classification et extraction d’informations d‘un document"
sidebar_name = "Présentation du projet"


def run():

    # TODO: choose between one of these GIFs
    #st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/1.gif")
    #st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/2.gif")
    st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/3.gif")

    st.title(title)

    st.markdown("---")

    st.markdown(
        """
        # Contexte

        ## Description : Déclaration d’un sinistre sur le chatbot du site d’un assureur.\n
        Certaines tâches peuvent être automatisées.\n
        #### Etapes 1:	Déclenchement du service de déclaration d’un sinistre suivant le texte tapé dans le chatbot.\n
        #### Etapes 2:	Vérification de l’identité de l’utilisateur :\n
        """)
    st.markdown("""
        * Le service demande un papier d’identité à l’utilisateur.\n
        * Le système vérifie le papier d’identité (Carte d’Identité ou Passeport).\n
        * Le système vérifie la date de validité de la pièce d’identité présentée.\n
        * Le système récupère les noms/prénoms/etc… au format requis par le LCBFT (Lutte Contre le Blanchiment et le Financement du Terrorisme)\n
        * L’assureur vérifie qu’il n’y a pas de soucis avec le registre correspondant.\n
        """)
    st.markdown("""
        #### Etapes 3:	Dépôt de la déclaration du sinistre et extraction des informations :\n
        * Le service demande le dépôt de la déclaration.\n
        * Le système vérifie le document. Est-ce bien le bon formulaire ?\n
        * Le système extrait les informations importantes du document et formate les données selon un certain modèle.
        """)

    st.markdown("---")

    st.markdown("""
        # Objectifs

        Nous nous sommes donnés comme objectif de classifier au mieux les documents du jeu de données fournis en mettant l’accent sur la classification des pièces d’identité (cartes d’identité et passeports).\n

        A travers cet objectif, nous avons cherché à explorer les diverses approches nous permettant la classification de documents, que ce soit par traitement du langage ou par Computer Vision.\n

        Enfin, nous avons cherché une piste réalisable pour extraire le texte des pièces d’identité.\n

        """
    )

"""
        A SUPPRIMER - CONSERVER POUR EXEMPLE
        
        Here is a bootsrap template for your DataScientest project, built with [Streamlit](https://streamlit.io).

        You can browse streamlit documentation and demos to get some inspiration:
        - Check out [streamlit.io](https://streamlit.io)
        - Jump into streamlit [documentation](https://docs.streamlit.io)
        - Use a neural net to [analyze the Udacity Self-driving Car Image
          Dataset] (https://github.com/streamlit/demo-self-driving)
        - Explore a [New York City rideshare dataset]
          (https://github.com/streamlit/demo-uber-nyc-pickups)"
"""
