from flask import Flask, render_template
import pandas as pd 
#from functions import *

def create_app():
    
    app = Flask(__name__)
    
    @app.route('/')
    @app.route('/home')
    def home_page():
        
        return render_template('home.html')
    
    # navigate urls
    @app.route("/get")
    def get():   
        return render_template("userID.html", keys = user_dict.keys())

    @app.route("/itemdict")
    def itemdict():
        return render_template("itemdict.html", item_dict = item_dict)

    
    @app.route("/discover")
    def discover():
        return render_template("discover.html", item_dict = item_dict)
    
    
    @app.route("/find")
    def find():
        return render_template("find.html", item_dict = item_dict)
        

    # URLs for inspecting stored data

    # ratings csv file
    @app.route("/ratings")
    def ratingsPrint():
        return render_template("csvtemplate.html", file = ratings.to_html())

    # item csv file
    @app.route("/items")
    def itemsPrint():
        return render_template("csvtemplate.html", file = items.to_html())

    # matrix print
    @app.route("/interactionmatrix")
    def matrixPrint():
        return render_template("csvtemplate.html", file = interactions.to_html())

    # URLs for usecases

    # Recommend Items to User
    
    @app.route("/get/<code>")
    def getItemtoUser(code):
        if code in user_dict:
            recitemsID, recitems, known_items = sample_recommendation_user(model = LFM_model, 
                                        interactions = interactions,user_id = code, 
                                        user_dict = user_dict,item_dict = item_dict, 
                                        threshold = 4,nrec_items = 10)
            
            if known_items == []:
                known_items.append("weren't able to find liked items")
            return render_template("getAfter.html", likes = known_items, recitems = recitems, recitemsID = recitemsID) ##statement + """<a href="/">Back to home page</a>"""
        else:
            return "Invalid user ID<p>" + """<a href="/">Back to home page</a>"""


    # Recommnend Users to an Item
    @app.route("/find/<code>")
    def findUser(code):
        if code in item_dict:
            user_id_list = sample_recommendation_item(model = LFM_model,
                                interactions = interactions,item_id = code,
                                user_dict = user_dict,item_dict = item_dict,
                                number_of_user = 15)
            
            
                
            statement = "Recommended Users:<br>"
            counter = 1
            for i in user_id_list:
                statement += str(counter) + '- ' + str(i) +"<br>"
                counter+=1
            statement += "<p>"
            return render_template("findAfter.html", user_list=user_id_list) #+ """<a href="/">Return to home page</a>"""    
        else:
            return "Invalid item ID<p>" + """<a href="/">Return to home page</a>"""
        
    # Similar items
    @app.route("/discover/<code>")
    def discoverItem(code):
        productArray = code.split(",") # stores item ids
        statement = "Items of interest :"
        one_dict = {}
        for itemID in productArray:
            one_score, one_name = item_item_recommendation(item_emdedding_distance_matrix = item_item_dist,
                                            item_id = itemID, item_dict = item_dict, n_items = 10)
            for i in range(0,10):
                if one_name[i] not in one_dict:
                    one_dict[one_name[i]] = one_score[i]
        
        one_dict_keys = sorted(one_dict, key=one_dict.get, reverse=True) # sorting keys
        
        for itemID in productArray:
            statement += " <i>{}</i> <b>|</b>".format(item_dict[itemID])
            
        statement += "<br> Item similar to the above item: <p>"
        counter = 1
        for i in range(0,10):
            statement += "{} - <b>{}</b> [{}]<br>".format(counter, item_dict[one_dict_keys[i]], one_dict[one_dict_keys[i]])
            
            counter+=1
            
        return statement + """<a href="/">Go back to home page</a>"""
        
        
    if __name__ == "__main__":
        app.run(debug=True)
        
        
# read in data
ratings = pd.read_csv("ratings.csv")
items = pd.read_csv("item.csv")     
"""
# making model
#1 creating interaction matrix
interactions = create_interaction_matrix(df = ratings,user_col = 'user_id',
item_col = 'item_id',rating_col = 'Rating')

#2 training model
LFM_model = runMF(interactions = interactions,n_components = 30,
loss = 'warp',epoch = 30,n_jobs = 4)

#3 creating dictionaries
user_dict, item_dict = create_dictionaries(interactions, items, "item_id", "Name")


#4 create item-item distance matrix
item_item_dist = create_item_emdedding_distance_matrix(model = LFM_model,
interactions = interactions) 
        """
        
create_app() #  runs the website