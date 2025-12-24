You are an expert in Python, Machine Learning, Multi-Modal applications and vector databases. 
Now I want to build up a system for the retrieval of nearly 1 million 3D assets. I will first describe the background, ideas and procedures. Then I need you to help me implement the system step by step.

Background: 
I have nearly 1 million 3D assets, each of them is associated with an asset ID and text caption. Each asset is rendered to different viewpoints under @data/gobjaverse. Notice that the subdirectory in @data/gobjaverse contains the gobjaverse instance id, you need to associate it with the objaverse instance ID via querying the dict from @data/gobjaverse_280k_index_to_objaverse.json. The text caption of objaverse (in english) is provided as json at @text_captions_cap3d.json.
Offline Embedding Build Up:
Generally you should first build up the embeddings of each embedding. Supported embedding modalities include english text, chinese text, single-image and multi-image. And we have two algorithm backends which support some of the modalities. The embeddings should then be stored in the vector database using pgvector. That's to say, a single 3D asset will be associated with a single text embedding vector, one or more image embedding vectors (conditioned on the embedding method we adopt).

Online App Deployment:
In practice the user will query the database using chinese text OR english text OR a SINGLE image. You should first embed corresponding query(using the SAME algoirthm with the one built up the database) and then search from the vector database. Notice that you need to support inner-modality query(query text using text or query the image using image) and cross-modality search(query image using text or query text using image, they will have the same length). Then the 3D asset with the best matched key will be returned to the user. You can assume that you will know the text is chinese or english upon request.

Step0: Convert the english captions json to another chinese version 
You can refer to @reference_qwen_batch.py to implement a standalone python script for submitting caption translation tasks. You should first split @text_captions_cap3d.json into multiple jsonl files with different customized ids (just like @test_model.jsonl) and then submit in a batch-wise manner.

Step1: Build up the text & image Embeddings 
There are two algorithms and I need you to support them both
(1) SigLip-based: it's relative simple, refer to @siglip_embed.py. You should convert the english caption into one embedding, and EVERY image into a single embedding. Note it doesn't support chinese captions.
(2) Qwen-API-based: the reference script is @qwen_embed.py. It embeds the english caption into one embedding, the chinese caption into one embedding , and embeds MULTIPLE (<= 8) images into ONE SINGLE embedding.

For text embeddings, you should refer to the @qwen_embed_text_batch.py, also doing the embedding in a batch-wise manner via first creating multiple .jsonl files.

Step2: Store the embeddings into database 
The database store should also support the two different methods (1 image embedding or multiple)
It should be based on pg vector.
Notice that embeddings from different algoirthm should be stored into different dbs, and english/chinese caption embeddings should be stored into different fields.

Step3: Build up a query service engine (backend)
Implement a minimal backend query service based on FastAPI and the stored vector database. 

Step4: Build up a minimal online demo (front-end)
Assuming that we have already built up a vector database, each 3D asset has an ID associated with it. I will provide you a BASE_URL, combined with which you can download the 3D model on-the-fly. Please use gradio to implement a minimal front-end application for the 3D asset retrieval. 
The user can enter text, upload image, configure the search mode they use (enable cross-modality search or not), and upon query finishes, you should download the 3D model and preview in the gradio 3D viewport.

Notice that you are an expert, please make good plans before you begin, ensure that you have good coding habits, organize the code structures well, and don't create too many markdown files. And you should allow setting up a maximum number of assets to process, for debug purposes.