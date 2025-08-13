# ML-Practice
A repo to practice machine learning 

For chapter 4s & sol - followed video by Data Professor:
https://www.youtube.com/watch?v=DctmeFx8s_k
which used the textbook Machine Learning with PyTorch and Scikit-Learn, Github:
https://github.com/rasbt/machine-learning-book

For voiceagent - followed video by Data Professor:
https://www.youtube.com/watch?v=cIQeIUIijbs

For voiceagent to run locally:
Go to Ollama website, download and install to local computer.
Command line "ollama" to check that it's installed.
Command line "ollama pull gemma3:1b" will download gemma onto your local computer.
Set API token: go to assemblyai.com and sign up. Login and go to the API key page. Here you can create new API keys. Then copy the API key that was generated. 
Command line: "vi ~/.bash_profile"
go to bottom line of bash script and type in:
export REPLICATE_API_TOKEN="r8_********"
export ASSEMBLY_API_KEY="*************"
Then to activate the bash script & the API token, command line "source ~/.bash_profile"
Install homebrew on your computer. Go to homebrew website and copy paste the command line on homebrew's website.
Command line: "brew install portsaudio" to deal with audio handling
Install XCode: command line: xcode-select --install (MACOS allowing to compiling code in command line)
Create conda environment: Command line: "conda create -n aai python=3.12" then "Y"
Command line: pip install assemblyai[extras]
Command line: pip install ollama
Command line: pip install replicate
Command line: pip install soundfile
Command line: pip install sounddevice

Activate script:
Open terminal and type "conda activate (environmentname)"
"source ~/.bash_profile"
"python (nameofscript).py"