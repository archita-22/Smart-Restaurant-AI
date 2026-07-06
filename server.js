import express from 'express';
import path from 'path';
import dotenv from 'dotenv';
import { ChatGoogleGenerativeAI } from '@langchain/google-genai';
import { createToolCallingAgent, AgentExecutor } from 'langchain/agents';
import { DynamicStructuredTool } from '@langchain/core/tools';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { z } from 'zod';

// Load environmental variables
dotenv.config();
const app = express();
const port = 3000;
app.use(express.json());

// 1. LLM (Gemini Model) Configuration
const model = new ChatGoogleGenerativeAI({
  model: "gemini-2.5-flash",
  maxOutputTokens: 2048,
  temperature: 0.7,
  apiKey: process.env.GOOGLE_API_KEY,
});

// 2. Menu Database
// Every dish carries: category, spice level, and how many people one plate serves.
// This is the single source of truth every tool below reads from.
const MENU = [
  // Breakfast
  { name: "Aloo Paratha", category: "breakfast", spice: "medium", serves: "1 person", diet: "veg" },
  { name: "Poha", category: "breakfast", spice: "mild", serves: "1-2 people", diet: "veg" },
  { name: "Masala Chai", category: "breakfast", spice: "mild", serves: "1 person", diet: "veg" },
  { name: "Chole Bhature", category: "breakfast", spice: "spicy", serves: "1-2 people", diet: "veg" },
  { name: "Idli Sambhar", category: "breakfast", spice: "mild", serves: "1 person", diet: "veg" },

  // Lunch
  { name: "Paneer Butter Masala", category: "lunch", spice: "mild", serves: "2-3 people", diet: "veg" },
  { name: "Dal Fry", category: "lunch", spice: "mild", serves: "2-3 people", diet: "veg" },
  { name: "Jeera Rice", category: "lunch", spice: "mild", serves: "2 people", diet: "veg" },
  { name: "Roti", category: "lunch", spice: "mild", serves: "1 person (2 pieces)", diet: "veg" },
  { name: "Chana Masala", category: "lunch", spice: "spicy", serves: "2-3 people", diet: "veg" },
  { name: "Veg Pulao", category: "lunch", spice: "medium", serves: "2 people", diet: "veg" },
  { name: "Egg Curry", category: "lunch", spice: "medium", serves: "2-3 people", diet: "non-veg" },

  // Dinner
  { name: "Veg Biryani", category: "dinner", spice: "medium", serves: "2-3 people", diet: "veg" },
  { name: "Raita", category: "dinner", spice: "mild", serves: "2-3 people", diet: "veg" },
  { name: "Salad", category: "dinner", spice: "mild", serves: "2-3 people", diet: "veg" },
  { name: "Butter Naan", category: "dinner", spice: "mild", serves: "1 person (2 pieces)", diet: "veg" },
  { name: "Malai Kofta", category: "dinner", spice: "mild", serves: "2-3 people", diet: "veg" },
  { name: "Butter Chicken", category: "dinner", spice: "medium", serves: "3-4 people", diet: "non-veg" },
  { name: "Chicken Biryani", category: "dinner", spice: "medium", serves: "3-4 people", diet: "non-veg" },
  { name: "Fish Fry", category: "dinner", spice: "spicy", serves: "2 people", diet: "non-veg" },

  // Mithai (desserts)
  { name: "Gulab Jamun", category: "mithai", spice: "mild", serves: "1 person (2 pieces)", diet: "veg" },
  { name: "Rasgulla", category: "mithai", spice: "mild", serves: "1 person (2 pieces)", diet: "veg" },
  { name: "Kaju Katli", category: "mithai", spice: "mild", serves: "2 people", diet: "veg" },
  { name: "Gajar Halwa", category: "mithai", spice: "mild", serves: "2-3 people", diet: "veg" },
  { name: "Rasmalai", category: "mithai", spice: "mild", serves: "1-2 people", diet: "veg" },
];

// Formats a list of dishes as one-per-line for readability in the chat UI
function formatDishList(items) {
  return items.map((d) => `• ${d.name}`).join("\n");
}

// Fuzzy, case-insensitive dish lookup
function findDish(query) {
  if (!query) return null;
  const q = query.toLowerCase().trim();
  return (
    MENU.find((d) => d.name.toLowerCase() === q) ||
    MENU.find((d) => d.name.toLowerCase().includes(q) || q.includes(d.name.toLowerCase())) ||
    null
  );
}

// 3. Tools

// 3a. Menu by category
const getMenuTool = new DynamicStructuredTool({
  name: "get_menu_tool",
  description: "Returns the list of dishes for a given category: breakfast, lunch, dinner, or mithai (desserts). Use this to answer 'what's in dinner/lunch/breakfast/mithai' style questions.",
  schema: z.object({
    category: z.string().describe("Food category, e.g., breakfast, lunch, dinner, mithai"),
  }),
  func: async ({ category }) => {
    const items = MENU.filter((d) => d.category === category.toLowerCase());
    if (items.length === 0) return `No menu found for "${category}".`;
    return `Here's what's in ${category}:\n${formatDishList(items)}`;
  },
});

// 3b. Recommend dishes by spice preference
const getSpiceRecommendationTool = new DynamicStructuredTool({
  name: "get_spice_recommendation_tool",
  description: "Recommends dishes matching a spice preference (spicy, medium, or mild/non-spicy). Optionally filter by category. Use this when the user says things like 'I want something spicy' or 'suggest a mild dish'.",
  schema: z.object({
    preference: z.enum(["spicy", "medium", "mild"]).describe("Requested spice level"),
    category: z.string().optional().describe("Optional category filter: breakfast, lunch, dinner, mithai"),
  }),
  func: async ({ preference, category }) => {
    let items = MENU.filter((d) => d.spice === preference);
    if (category) items = items.filter((d) => d.category === category.toLowerCase());
    if (items.length === 0) return `No ${preference} dishes found${category ? ` in ${category}` : ""}.`;
    return items.map((d) => d.name).join(", ");
  },
});

// 3c. Spice level of a single named dish
const getSpiceLevelTool = new DynamicStructuredTool({
  name: "get_spice_level_tool",
  description: "Returns the spice level (spicy, medium, or mild) of one specific named dish. Use this whenever the user asks 'is X spicy?' or 'how spicy is X?' about a single dish.",
  schema: z.object({
    dishName: z.string().describe("The dish name to check the spice level for"),
  }),
  func: async ({ dishName }) => {
    const dish = findDish(dishName);
    if (!dish) return `Couldn't find "${dishName}" on the menu.`;
    return `${dish.name} is ${dish.spice}.`;
  },
});

// 3d. Filter menu by diet type (veg / non-veg)
const getDietMenuTool = new DynamicStructuredTool({
  name: "get_diet_menu_tool",
  description: "Lists dishes matching a diet type: 'veg' or 'non-veg'. Optionally filter by category (breakfast, lunch, dinner, mithai). Use this for questions like 'is there any non-veg?', 'what non-veg do you have?', or 'show me veg options in dinner'.",
  schema: z.object({
    dietType: z.enum(["veg", "non-veg"]).describe("Diet type to filter by"),
    category: z.string().optional().describe("Optional category filter: breakfast, lunch, dinner, mithai"),
  }),
  func: async ({ dietType, category }) => {
    let items = MENU.filter((d) => d.diet === dietType);
    if (category) items = items.filter((d) => d.category === category.toLowerCase());
    if (items.length === 0) {
      return `No ${dietType} dishes found${category ? ` in ${category}` : ""}.`;
    }
    return `Here's what's ${dietType}${category ? ` in ${category}` : ""}:\n${formatDishList(items)}`;
  },
});

// 3e. Compare spice level between two named dishes
const compareSpiceTool = new DynamicStructuredTool({
  name: "compare_spice_tool",
  description: "Compares the spice level of two specific dishes by name and states which one is spicier, or if they're the same. Use this for questions like 'which is spicier, Chole Bhature or Dal Fry?' or 'X vs Y, which is better if I don't like spicy food?'.",
  schema: z.object({
    dishA: z.string().describe("First dish name"),
    dishB: z.string().describe("Second dish name"),
  }),
  func: async ({ dishA, dishB }) => {
    const a = findDish(dishA);
    const b = findDish(dishB);
    if (!a) return `Couldn't find "${dishA}" on the menu.`;
    if (!b) return `Couldn't find "${dishB}" on the menu.`;
    const order = { mild: 0, medium: 1, spicy: 2 };
    if (order[a.spice] === order[b.spice]) {
      return `${a.name} and ${b.name} are both ${a.spice}, so they're about the same in heat.`;
    }
    const spicier = order[a.spice] > order[b.spice] ? a : b;
    const milder = spicier === a ? b : a;
    return `${spicier.name} (${spicier.spice}) is spicier than ${milder.name} (${milder.spice}). Go with ${milder.name} if you'd prefer something gentler, or ${spicier.name} if you want more heat.`;
  },
});

// 3d. Serving size / how many people a plate feeds
const getServingSizeTool = new DynamicStructuredTool({
  name: "get_serving_size_tool",
  description: "Returns how many people one plate/order of a specific named dish typically serves. Use this for questions like 'how many people can 1 plate of Butter Chicken feed?' or 'is one Veg Biryani enough for 2 of us?'.",
  schema: z.object({
    dishName: z.string().describe("The dish name to look up serving size for"),
  }),
  func: async ({ dishName }) => {
    const dish = findDish(dishName);
    if (!dish) return `Couldn't find "${dishName}" on the menu.`;
    return `One plate of ${dish.name} typically serves ${dish.serves}.`;
  },
});

const tools = [
  getMenuTool,
  getSpiceRecommendationTool,
  getSpiceLevelTool,
  getDietMenuTool,
  compareSpiceTool,
  getServingSizeTool,
];

// 4. System and Chat Prompt Engineering Template
const prompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    "You are REStro's friendly restaurant assistant. You have six tools: " +
    "get_menu_tool (list dishes by category: breakfast, lunch, dinner, mithai), " +
    "get_spice_recommendation_tool (suggest dishes by spice level), " +
    "get_spice_level_tool (spice level of ONE named dish), " +
    "get_diet_menu_tool (list veg or non-veg dishes, optionally by category), " +
    "compare_spice_tool (compare spice level between two named dishes), " +
    "and get_serving_size_tool (how many people one plate feeds). " +
    "Rules to follow strictly: " +
    "1) If the user asks a generic question like 'what's on the menu' without naming a specific meal, " +
    "do NOT call a tool yet — first ask which meal they mean: breakfast, lunch, dinner, or mithai. " +
    "Once they name one, call get_menu_tool for it. " +
    "2) If the user asks about non-veg or veg options (e.g. 'is there any non-veg?'), use get_diet_menu_tool " +
    "instead of get_menu_tool. " +
    "3) If the user asks whether ONE specific dish is spicy, use get_spice_level_tool, not compare_spice_tool. " +
    "4) Always use a tool instead of guessing, and never invent dishes that weren't returned by a tool. " +
    "5) When a tool result already contains a bulleted, line-broken list, paste that formatting into your reply " +
    "exactly as given (one dish per line) — do not collapse it into a comma-separated sentence. " +
    "Keep the rest of your tone short, warm, and conversational.",
  ],
  ["human", "{input}"],
  ["placeholder", "{agent_scratchpad}"],
]);

// 5. Orchestrate the Agent and Executor Strategy
const agent = await createToolCallingAgent({
  llm: model,
  tools,
  prompt,
});
const executor = new AgentExecutor({
  agent,
  tools,
  verbose: true, // Enables step-by-step reasoning logs in your terminal
  maxIterations: 3, // Needs enough room for: call tool -> read result -> compose final answer
  returnIntermediateSteps: true,
});

// Serve frontend client UI webpage
const __dirname = path.resolve();
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Restaurant Chat API Endpoint
app.post('/api/chat', async (req, res) => {
  const userInput = req.body.input;
  console.log("User Input:", userInput);
  try {
    const response = await executor.invoke({ input: userInput });
    console.log("Agent Full Response:", response);

    if (response.output && !response.output.includes("Agent stopped due to max iterations")) {
      return res.json({ output: response.output });
    } else if (response.intermediateSteps && response.intermediateSteps.length > 0) {
      const stepData = response.intermediateSteps[response.intermediateSteps.length - 1].observation;
      return res.json({ output: stepData });
    }
    res.status(404).json({ output: "Sorry, the assistant could not find an answer." });
  } catch (error) {
    console.error("Error during execution of the agent:", error);
    res.status(500).json({ output: "Sorry, something went wrong. Please try again later." });
  }
});

app.listen(port, () => {
  console.log(`Server is running at http://localhost:${port}`);
});
