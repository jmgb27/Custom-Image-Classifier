<div class="markdown prose w-full break-words dark:prose-invert dark"><h1>README</h1><p>This project is a image classifier that uses pre-trained models such as VGG16 and Resnet18 to classify images of different classes. The user can specify the architecture of the model, the learning rate, the number of hidden units, the number of training epochs, and whether to use GPU for training when running the script.</p><h2>Dependencies</h2><ul><li>torch</li><li>torchvision</li><li>argparse</li><li>matplotlib</li><li>numpy</li><li>PIL</li></ul><h2>Usage</h2><p>The script can be run using the following command:</p><pre><div class="bg-black mb-4 rounded-md"><div class="flex items-center relative text-gray-200 bg-gray-800 px-4 py-2 text-xs font-sans"><button class="flex ml-auto gap-2"><svg stroke="currentColor" fill="none" stroke-width="2" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4" height="1em" width="1em" xmlns="http://www.w3.org/2000/svg"><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path><rect x="8" y="2" width="8" height="4" rx="1" ry="1"></rect></svg></button></div><div class="p-4 overflow-y-auto"><code class="!whitespace-pre-wrap hljs language-css">python <span class="hljs-selector-tag">train</span><span class="hljs-selector-class">.py</span> dataset_folder 
</code></div></div></pre><ul><li><code>dataset_folder</code>: directory containing the training data</li><li><code>save_dir</code> (optional): directory to save checkpoints</li><li><code>arch</code> (optional): model architecture, can be either "vgg16" or "resnet18"</li><li><code>learning_rate</code> (optional): learning rate for the optimizer</li><li><code>hidden_units</code> (optional): number of hidden units for the classifier</li><li><code>epochs</code> (optional): number of training epochs</li><li><code>gpu</code> (optional): flag to use GPU for training</li></ul><h2>Data Preparation</h2><p>The script expects the dataset to be structured as follows:</p><pre><div class="bg-black mb-4 rounded-md"><div class="flex items-center relative text-gray-200 bg-gray-800 px-4 py-2 text-xs font-sans"><button class="flex ml-auto gap-2"><svg stroke="currentColor" fill="none" stroke-width="2" viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round" class="h-4 w-4" height="1em" width="1em" xmlns="http://www.w3.org/2000/svg"><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path><rect x="8" y="2" width="8" height="4" rx="1" ry="1"></rect></svg></button></div><div class="p-4 overflow-y-auto"><code class="!whitespace-pre-wrap hljs language-bash">dataset_folder
|
|__ train
|   |
|   |__ class1
|   |   |
|   |   |__ image1.jpg
|   |   |__ image2.jpg
|   |   |__ ...
|   |
|   |__ class2
|   |   |
|   |   |__ image1.jpg
|   |   |__ image2.jpg
|   |   |__ ...
|   |
|   |__ ...
|
|__ valid
|   |
|   |__ class1
|   |   |
|   |   |__ image1.jpg
|   |   |__ image2.jpg
|   |   |__ ...
|   |
|   |__ class2
|   |   |
|   |   |__ image1.jpg
|   |   |__ image2.jpg
|   |   |__ ...
|   |
|   |__ ...
|
|__ <span class="hljs-built_in">test</span>
    |
    |__ class1
    |   |
    |   |__ image1.jpg
    |   |__ image2.jpg
    |   |__ ...
    |
    |__ class2
    |   |
    |   |__ image1.jpg
    |   |__ image2.jpg
    |   |__ ...
    |
    |__ ...
</code></div></div></pre><h2>Output</h2><p>The script will output the number of images in each dataset and the classes. It will also save the trained model to the specified <code>save_dir</code> with the name <code>checkpoint.pth</code>.</p><p>It will also display the loss and accuracy of the model after each epoch.</p>
