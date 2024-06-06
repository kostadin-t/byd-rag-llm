const express = require("express");
const app = express();
const OpenAI = require("openai");
const { createClient } = require("@supabase/supabase-js");
const { RecursiveCharacterTextSplitter } = require("langchain/text_splitter");
const { PDFLoader } = require("langchain/document_loaders/fs/pdf");
require("dotenv").config();

app.use(express.json());

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_ANON_KEY
);

app.post("/embed", async (req, res) => {
  try {
    await generateAndStoreEmbeddings();
    res.status(200).json({ message: "Successfully Embedded" });
  } catch (error) {
    console.log(error);
    res.status(500).json({
      message: "Error occurred",
    });
  }
});

app.post("/query", async (req, res) => {
  try {
    const { query } = req.body;
    const result = await handleQuery(query);
    res.status(200).json(result);
  } catch (error) {
    console.log(error);
    res.status(500).json({
      message: `Error occurred ${error.message}`,
    });
  }
});

async function generateAndStoreEmbeddings() {
  const loader = new PDFLoader("bydprojects.pdf");
  const docs = await loader.load();
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 2048,
    chunkOverlap: 200,
  });

  const chunks = await textSplitter.splitDocuments(docs);
  const promises = chunks.map(async (chunk) => {
    const cleanChunk = chunk.pageContent.replace(/\n/g, " ").replace(/\u0000/g, '');

    const embeddingResponse = await openai.embeddings.create({
      model: "text-embedding-3-small",
      input: cleanChunk,
    });

    const [{ embedding }] = embeddingResponse.data;
    const cleanEmbedding = JSON.stringify(embedding).replace(/\u0000/g, '');

    const { data, error } = await supabase.from("documents").insert([
      {
        content: cleanChunk,
        embedding: JSON.parse(cleanEmbedding),
    },
    ]);

    if (error) {
      console.log(
        "error",
        error.message,
        error.code,
        error.details,
        error.hint
      );
      throw error;
    }
  });

  await Promise.all(promises);
}

async function handleQuery(query) {
  const input = query.replace(/\n/g, " ");

  const embeddingResponse = await openai.embeddings.create({
    model: "text-embedding-3-small",
    input,
  });

  const [{ embedding }] = embeddingResponse.data;

  const { data: documents, error } = await supabase.rpc("match_documents", {
    query_embedding: embedding,
    match_threshold: 0.5,
    match_count: 10,
  });

  if (error) throw error;

  let contextText = "";

  contextText += documents
    .map((document) => `${document.content.trim()}---\n`)
    .join("");

  const messages = [
    {
      role: "system",
      content: `You are a representative that is very helpful when it comes to talking about SAP Business ByDesign, Only ever answer
      truthfully and be as helpful as you can!`,
    },
    {
      role: "user",
      content: `Context sections: "${contextText}" Question: "${query}" Answer as simple text:`,
    },
  ];

  const completion = await openai.chat.completions.create({
    messages,
    model: "gpt-4",
    temperature: 0.8,
  });

  return completion.choices[0].message.content;
}

app.listen("3035", () => {
  console.log("App is running on port 3035");
});
