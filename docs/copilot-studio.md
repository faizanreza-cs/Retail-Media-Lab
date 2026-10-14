# Use the Ads Copilot from Microsoft Copilot Studio

The API's `/ask` and `/optimize` endpoints can be added to a Copilot Studio agent as actions
through a Power Platform custom connector. Teams then ask budget and performance questions in
Teams or Microsoft 365 Copilot, and the answers come from this project's tools.

## 1. Deploy the API

Run the Docker image anywhere with HTTPS (for example Azure Container Apps or App Service):

```bash
docker build -t retail-media-lab .
docker run -p 8000:8000 \
  -e RML_API_TOKEN=<a long random key> \
  -e RML_PROVIDER=azure -e RML_MODEL=<your Azure OpenAI deployment> \
  -e AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com -e AZURE_OPENAI_API_KEY=<key> \
  retail-media-lab
```

With `RML_API_TOKEN` set, every request must send it in the `X-API-Key` header. Without an LLM
configured, `/ask` still works in the offline mock mode.

## 2. Create the custom connector

1. In [`copilot/ads-copilot.swagger.json`](../copilot/ads-copilot.swagger.json), set `host` to
   your API's host name.
2. In Power Apps or Power Automate, go to **Custom connectors**, choose **Import an OpenAPI
   file** and upload that file. (It is OpenAPI 2.0, which the custom connector wizard accepts.)
3. On the **Security** step choose **API Key**, header `X-API-Key`.
4. Create the connector, then use **Test** with a question such as "How should we reallocate
   the budget?" and your key.

## 3. Add it to a Copilot Studio agent

1. Open your agent in Copilot Studio and add a tool (action) from the custom connector:
   **AskAdsCopilot**, and **RecommendBudget** if you want plans with explicit limits.
2. Describe when to use it, for example: "Use for questions about Amazon Ads performance,
   ROAS, attribution, incrementality tests and budget allocation."
3. Test in the test pane, then publish to Teams.

Menu names change between Power Platform releases; see Microsoft's documentation for
[custom connectors](https://learn.microsoft.com/en-us/connectors/custom-connectors/) and
[Copilot Studio](https://learn.microsoft.com/en-us/microsoft-copilot-studio/) if a step looks
different.

## Security notes

- Keep the key in the connector's connection, never in the OpenAPI file. Rotate it like any
  other secret.
- The API is read-only: it can't change campaigns or budgets, only recommend them.
- SQL from the agent is checked and runs read-only with a timeout, and shopper ids can only be
  counted, grouped or joined on, never returned (see the README's Guardrails for the limits).
