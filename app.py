from flask import Flask, request, make_response, jsonify
from flask_restful import Resource, Api
from flask_sqlalchemy import SQLAlchemy
import psycopg2
from object_matcher import ObjectRegistration, ObjectMatching
# Create the app 
app = Flask(__name__)
# Configure the postgres database
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:postgres@localhost:5432/AIOBJECTS'
# Create the database object
db = SQLAlchemy(app)

# Create the table
class ObjectData(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    object_name = db.Column(db.String(100), nullable=False)
    object_data = db.Column(db.Text, nullable=False)

# Create the api
api = Api(app)

# Create the resource
class ObjectRegister(Resource):
    def post(self):
        jsonData = request.json
        # Get the image data
        imageData = jsonData['imageData']
        # Get the object name
        objectName = jsonData['objectName']
        # Register the object
        objectData = ObjectRegistration(imageData)
        # Check if the object is registered
        if objectData['status']:
            # Create the object
            object = ObjectData(object_name=objectName, object_data=objectData['data'])
            # Add the object to the database
            db.session.add(object)
            # Commit the changes
            db.session.commit()
            # Return the message
            return make_response(jsonify({'message':objectData['message']}), 200)
        else:
            # Return the message
            return make_response(jsonify({'message':objectData['message']}), 400)
        
class ObjectMatch(Resource):
    def post(self):
        jsonData = request.json
        # Get the image data
        imageData = jsonData['imageData']
        # Get the object data from the database
        objectData = ObjectData.query.all()

        closeMatches = []
        for data in objectData:
            # Match the object
            objectMatch = ObjectMatching(imageData, data.object_data)

            if not objectMatch['flag']:
                return make_response(jsonify({'message':objectMatch['message']}), 500)
            
            if objectMatch['status']:
                closeMatches.append({'objectId':data.id, 'objectName':data.object_name, 'objectData':data.object_data , 'matches':objectMatch['matches']})
        # Sort the matches in descending order
        closeMatches.sort(key=lambda x: x['matches'], reverse=True)
        # print(closeMatches)
        if len(closeMatches) > 0 and len(closeMatches) < 3:
            return make_response(jsonify({'message':'Matches Found','data':closeMatches}), 200)
        elif len(closeMatches) >= 3:
            return make_response(jsonify({'message':'Matches Found','data':closeMatches[:3]}), 200)
        else:
            return make_response(jsonify({'message':'No Matches Found'}), 400)

class ObjectView(Resource):
    def get(self):
        # Get the object data from the database
        object_data = ObjectData.query.all()
        data = []
        # Check if the object data is empty
        if len(object_data) == 0:
            # Return the message
            return make_response(jsonify({'message':'No Object Data'}), 400)
        else:
            # Return the object data
            for object in object_data:
                data.append({'objectId':object.id, 'objectName':object.object_name, 'objectData':object.object_data})
            return make_response(jsonify({'message':'Object Data Found','data':data}), 200)

# Add the resource to the api
api.add_resource(ObjectRegister, '/register')
api.add_resource(ObjectMatch, '/match')
api.add_resource(ObjectView, '/data')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

# >>> flask shell
# >>> from yourapplication import db
# >>> db.create_all()