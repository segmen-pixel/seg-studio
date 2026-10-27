# AI support with Ollama: a walk-through for Windows and macOS

[AI support](ai-support.md) lets a vision model label a project for you. This
page goes through one way of setting it up, from a machine that has
Seg-Studio and no model server to a finished run: Ollama as the model server,
on the same machine as Seg-Studio. It is an example and not the only way. The
other model servers, a server on another machine and the MCP bridge itself
are in [AI support and the MCP bridge](ai-support.md).

Ollama and the model are separate software. Seg-Studio ships neither, and
which model you run is your choice. Each has its own licence: read Ollama's,
and the model's (`ollama show <model> --license`), against the use you have
in mind.

## What the steps were checked on

| | Windows | macOS |
|---|---|---|
| Machine | Windows 11, NVIDIA GPU with 24 GB of memory | macOS 26, Apple silicon with 64 GB of memory |
| Ollama | 0.33.3 | 0.32.14 |
| Model | `qwen3.8:27b` | `qwen3.8:27b-mlx` |
| Model: download | 18 GB | 18 GB |
| Model: memory while loaded (`ollama ps`) | 17 GB, all on the GPU | 21 GB during the panel's run, up to 34 GB during the chat page |

Any model that passes the check in step 2 is used the same way. A smaller
model fits a smaller machine but may not get as far: in the same check on
macOS, a model of about 9 billion parameters kept reading the state of the
project and labelled nothing.

## Before you start

- **Seg-Studio is installed and starts** ([Installation in detail](install.md)).
  Nothing is added to it for this page: the MCP library the panel uses is
  part of the install.
- **A project with some images labelled by hand and others not yet.** The
  model takes the labelled images as its examples.
- **An internet connection for the downloads**: Ollama, the model and, on
  macOS, the SAM checkpoints the first time they are used. The labelling
  itself runs on the machine. With Ollama on the same machine and a model
  that runs on it, your images do not leave it. A model that Ollama hands on
  to a cloud service (its `cloud` models) receives them: see
  [What the model is told about your data](ai-support.md#what-the-model-is-told-about-your-data).

## 1. Install Ollama

Download the installer from <https://ollama.com/download> and install it the
way Ollama describes for [Windows](https://docs.ollama.com/windows) or
[macOS](https://docs.ollama.com/macos). From then on it runs in the
background and answers at `http://127.0.0.1:11434`, which is where Seg-Studio
looks for it.

Check it in a new terminal (Command Prompt or PowerShell on Windows, Terminal
on macOS):

```
ollama --version
```

Then open `http://127.0.0.1:11434` in a browser. The page says
`Ollama is running`.

## 2. Get a model that sees images and calls tools

```
ollama pull qwen3.8:27b
```

On Apple silicon, pull `qwen3.8:27b-mlx` instead; the check used it. Either
is an 18 GB download. Then look at what the model can do, under the name you
pulled (`ollama show qwen3.8:27b-mlx` on macOS):

```
ollama show qwen3.8:27b
```

```
  Capabilities
    completion
    vision
    tools
    thinking
```

`vision` and `tools` must both be in that list. A model without either
cannot label. The panel lists it all the same, because it lists every model
Ollama holds (`ollama list` prints the same names).

## 3. Open the AI support panel

1. Start Seg-Studio (`start-windows.bat`, or `bash start-macos.sh`).
2. Open the project on the **Annotate** screen and press **AI** at the foot of
   the tool column.
3. The top of the panel reads **Ollama**, and the list beside it holds the
   models from step 2. Nothing was set up for this: with nothing configured,
   Seg-Studio looks for Ollama on its own machine. Choose your model in the
   list. Seg-Studio remembers it.

## 4. Start a run

1. Type what to label, in your own words and with how to judge it, and press
   **Start**. The checks used this sentence, in Japanese:

   > Label the defects (small dark dots and scratches) in the unlabelled
   > images, the way the labelled ones show.

   The badge at the top of the panel changes to **Working**.
2. On the first run in a project, the model rehearses on images you labelled
   and settles the zoom at which it finds your objects. No mask is saved
   during this; the zoom it settles on is kept for the project, and later
   runs reuse it. In the macOS check it took about four minutes, the download
   of the SAM checkpoints included.
3. The panel then asks whether to go ahead, with the number of images left
   and an estimate (20 images and about 20 minutes in the macOS check). Press
   **Proceed at ...**. With no answer for 20 minutes the run ends and nothing
   is labelled. A project labelled as a scattering of small marks has no
   rehearsal and asks with a plain **Proceed**; the User Guide describes
   both.
4. It goes through the unlabelled images one at a time. A mask it writes
   shows in the image list. An image on which it writes nothing, or whose
   masks it is unsure about, gets a **Review** flag for you to look at. In
   the macOS check the 20 images took about 13 minutes: masks on 5 of them, a
   flag on the other 15. The whole run, rehearsal included, took about 17
   minutes and ended by itself with a summary of what it did.
5. **Stop** ends the run at the end of the step it is in. While the model is
   answering, that is when its answer arrives (about half a minute in the
   check).

What the run wrote is recorded as the AI's work, and it does not replace a
mask a person drew. Every control of the panel is described in the
[User Guide](user-guide.md#ai-support-panel).

## 5. The same model in the chat page (optional)

`scripts/examples/qwen_mcp_chat.py` is a small chat page over the same
bridge: you type, the model answers and uses Seg-Studio's tools, and each
tool call is shown. It is an example and not part of the product. It has no
login, it listens on this machine only, and its bridge may write to any
project ([SECURITY.md](../SECURITY.md#the-example-chat-page)).

With Seg-Studio running, open another terminal in the Seg-Studio folder.

Windows:

```
.venv-windows\Scripts\python.exe scripts\examples\qwen_mcp_chat.py --backend ollama --model qwen3.8:27b --lang en
```

macOS:

```
.venv-macos/bin/python scripts/examples/qwen_mcp_chat.py --backend ollama --model qwen3.8:27b-mlx --lang en
```

The terminal stays quiet until you send something. Open
`http://127.0.0.1:8765`. The top of the page names the model and
`ollama · http://127.0.0.1:11434`. Start with something that only reads, and
name the project (put your project's name in place of `my-project`):

> Project my-project: count the images that have no label yet. Do not write
> anything.

In the checks the model answered with the number and the file names after
one tool call. Stop the page with `Ctrl+C` in its terminal. Without
`--lang en` the page and the model's replies are in Japanese.

## When a step does not go as written

| What you see | What to do |
|---|---|
| The panel's model list is empty | The panel shows the address it asked and what came back. Check that `http://127.0.0.1:11434` says `Ollama is running` and that `ollama list` names a model, both on the machine that runs Seg-Studio |
| The run keeps reading the state of the project and labels nothing | The model is not up to the job. Press **Stop** and choose a larger model |
| The first answer is a long time coming | Ollama is loading the model. `ollama ps` shows what is loaded, how much memory it takes and whether it is on the GPU (`100% GPU`) |
| Ollama is on another machine | That is set on the machine that runs Seg-Studio, not in the panel: [AI support, section A](ai-support.md#a-the-ai-support-panel) |

Ollama's own chat window and `ollama run` talk to the model directly;
Seg-Studio's tools are not connected to them (checked with Ollama 0.33.3). To
have a model work on a project, use the panel, the chat page above, or an MCP
client of your own
([AI support, section B](ai-support.md#b-your-own-mcp-client)).
