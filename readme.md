# An LLM-Based conversational agent for browsing through Citizen Science resources

## A functional prototype for aspiring citizen scientists to learn more about water-related citizen science methods and best practices!

This app was built as a part of my Masters' internship with [PULSAQUA](https://www.pulsaqua.com/). It has access to a list of curated resources about water-related citizen science, including general best practices and specific methods. It is built using the [Streamlit](https://streamlit.io/) and [LangGraph](https://www.langchain.com/langgraph) python packages for the user interface and general architecture. In addition, it connects to a [Qdrant](https://qdrant.tech/) vector store database, the [google GenAI](https://cloud.google.com/ai/generative-ai) [Gemini-2.0-flash model](https://deepmind.google/models/gemini/flash/), and the [openAI](https://openai.com/api/) text-embedding-3-small model for operation. 

## How to use this app?

You can access a live version of the app at [this link](https://csadvisor-3ksupkpvxwspnxpucskewm.streamlit.app/), or clone the repository to host your own version (see the [streamlit website](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app) for more information).

## How does it work?

The app uses a basic Retrieval Augmented Generation structure, where the user question is first used to search a vector database containing the curated citizen science resources, and the three most relevant documents are returned. Afterwards, the gemini LLM is prompted with both the retrieved documents and the user question to provide an answer based on the documents. If the question cannot be answered from the documents, the LLM is instructed to give its best guess while explicitly stating that it is guessing. Advanced features like web searching and conversation memory are not yet implemented, but they should be relatively easy to implement into the existing infrastructure.

For more technical information about RAG systems, see the [wikipedia](https://en.wikipedia.org/wiki/Retrieval-augmented_generation) page or [this tutorial](https://qdrant.tech/documentation/agentic-rag-langgraph/) by Qdrant.

## Can I adapt it for another subject?

This general setup can be replicated for any other subject matter. Just swap out the qdrant vector store to one containing your database of subject-relevant resources and it's ready to go!

## How can it be improved?

I have wrapped up my internship, meaning that I will not be actively working on this app for the forseeable future. If you are keen to improve on this work, I have the following suggestions (ranked in order of my intuitive guess on the effort/reward ratio):

**Use a better model** 

The current gemini-2.0-flash is a relatively cheap model, and I expect performance to improve when using more expensive and advanced models such as [gemini-2.5-pro-preview](https://ai.google.dev/gemini-api/docs/models#gemini-2.5-pro-preview-06-05). Alternatively, you could consider testing other companies' state-of-the-art models, such as [Claude opus 4](https://docs.anthropic.com/en/docs/about-claude/models/overview) or [GPT-4.1](https://platform.openai.com/docs/models). This option is relatively easy to implement, as it only involves swapping out the model connection and API.

**Add additional functionality** 

Adding a web-searching option and conversational memory requires a bit of effort, but luckily there are [good tutorials](https://medium.com/@ipeksahbazoglu/building-a-multi-tool-agent-with-langgraph-and-google-vertex-ai-e37aa6d41265) on how to do this. I expect this method to significantly improve the utility, as there is a lot of information about citizen science on the internet that can be used to supplement the resource database. However, there should be some logic in place to appropriately prioritize the curated resources over web information. In addition, [conversation memory](https://python.langchain.com/docs/how_to/chatbots_memory/) could improve the user experience to make the chatbot more useful.

**Tweak the system prompt.** 

[Improving the system prompt](https://blog.promptlayer.com/system-prompt-vs-user-prompt-a-comprehensive-guide-for-ai-prompts/) is pretty easy as it just involves changing some text, and it can  lead to notable improvements in performance. These improvements likely won't be as big as the previous two suggestions, but it is still a worthwile option to explore. This does require a good system for A/B testing different versions of the system prompt.

**Add more resources to the database**. 

Expanding the database can result in significant improvements, but is also the most difficult to do. I created the initial database using [unstructured.io](https://unstructured.io/). This is a convenient but expensive service for this purpose. There is a generous 14-day free trial, but for longer-term use and maintenance I suggest replicating their workflow manually to save costs. There are [some tutorials](https://medium.com/@aminajavaid30/building-a-rag-system-the-data-ingestion-pipeline-d04235fd17ea) for this purpose. If you are pursuing this option, I also suggest using a more robust text embedding model. The current version uses openAI's text-embedding-3-small, but the text-embedding-3-large could yield [better performance](https://platform.openai.com/docs/guides/embeddings). This will be slightly more expensive, but the embedding model is not the major source of cost in this app (the LLM calls contain many more tokens).

**More fine-grained search options** 

You could add checkboxes or drop-down menus in the streamlit interface for users to indicate if they want resources in a specific language, or want to only search for currently active projects (for example). This could make searching for specific resources easier, adding a lot of functionality. However, this would also take a significant amount of effort to implement, as there is currently no easy way to filter the resources by metadata. Implementing this feature would likely require manually categorizing the resource database and manually implementing a more sophisticated search logic. A less sophisticated implementation could be to simply add some extra instructions to the LLM system prompt and/or user question based on the selected sliders, but I don't forsee this adding a lot of functionality. Therefore, I suggest exploring this option only after the previous suggestions have been implemented.