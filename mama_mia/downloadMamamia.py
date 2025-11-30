import synapseclient
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("SYNAPSE")

syn = synapseclient.Synapse()
syn.login(authToken=api_key)
entity = syn.get("syn60868042")