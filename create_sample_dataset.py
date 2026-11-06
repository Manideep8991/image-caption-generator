import os
from PIL import Image, ImageDraw, ImageFont

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")
images_dir = os.path.join(data_dir, "Images")
os.makedirs(images_dir, exist_ok=True)

samples = [
    {
        "id": "1000268201_693b08cb0e.jpg",
        "bg_top": (135, 206, 235),
        "bg_bot": (34, 139, 34),
        "elements": [
            ("rect", (40, 160, 120, 290), (220, 20, 60)),     # Playground slide
            ("rect", (160, 120, 260, 290), (200, 100, 50)),   # Climbing frame
            ("circle", (90, 120, 30), (255, 218, 185)),       # Child head
            ("rect", (75, 150, 105, 220), (255, 105, 180)),   # Pink dress
        ],
        "label": "Child on Playground Slide",
        "captions": [
            "A child in a pink dress is climbing up a set of stairs in an entry way",
            "A little girl in a pink dress goes down a slide on a sunny playground",
            "A young girl in bright clothing plays outside on playground equipment",
            "A child wearing pink is having fun playing on a colorful playground",
            "A little girl smiles as she climbs the steps of a playground slide"
        ]
    },
    {
        "id": "1001773457_577c3a7d70.jpg",
        "bg_top": (100, 149, 237),
        "bg_bot": (30, 144, 255),
        "elements": [
            ("circle", (150, 170, 45), (139, 69, 19)),       # Brown dog body
            ("circle", (195, 145, 25), (139, 69, 19)),       # Dog head
            ("circle", (210, 150, 10), (0, 0, 0)),           # Snout
            ("circle", (130, 200, 15), (255, 255, 255)),     # Water splash
            ("circle", (170, 210, 20), (255, 255, 255)),     # Water splash
        ],
        "label": "Dog Splashing in Water",
        "captions": [
            "A brown dog is running and splashing through shallow river water",
            "A wet brown dog happily plays in a body of clear water",
            "A dog jumps in the lake creating white splashes all around",
            "A medium brown dog bounds through cool water on a warm day",
            "A playful canine enjoys a refreshing swim and splash outdoors"
        ]
    },
    {
        "id": "1002674143_1b742ab4b8.jpg",
        "bg_top": (70, 130, 180),
        "bg_bot": (160, 82, 45),
        "elements": [
            ("circle", (90, 230, 35), (40, 40, 40)),         # Bike wheel back
            ("circle", (210, 230, 35), (40, 40, 40)),        # Bike wheel front
            ("rect", (110, 180, 190, 195), (255, 69, 0)),    # Bike frame
            ("circle", (150, 120, 22), (255, 218, 185)),     # Biker head / helmet
            ("rect", (140, 145, 165, 195), (50, 205, 50)),   # Green jersey
        ],
        "label": "Biker on Mountain Trail",
        "captions": [
            "A cyclist in a green shirt rides a mountain bicycle along a dirt trail",
            "A mountain biker navigates a rugged rocky path with trees in the background",
            "A person wearing a helmet speeds down an unpaved mountain trail on a bike",
            "An athletic cyclist rides outdoors through a scenic dirt path",
            "A cyclist rides a mountain bike along a winding wilderness dirt track"
        ]
    },
    {
        "id": "1003163366_44323f5815.jpg",
        "bg_top": (112, 128, 144),
        "bg_bot": (105, 105, 105),
        "elements": [
            ("rect", (30, 40, 90, 250), (169, 169, 169)),    # Building 1
            ("rect", (210, 20, 270, 250), (128, 128, 128)),  # Building 2
            ("circle", (150, 100, 25), (245, 222, 179)),     # Musician head
            ("rect", (135, 130, 165, 210), (25, 25, 112)),   # Dark blue jacket
            ("circle", (160, 175, 30), (210, 105, 30)),      # Acoustic guitar
        ],
        "label": "Man Playing Guitar on Street",
        "captions": [
            "A street musician wearing a jacket plays acoustic guitar on a sidewalk",
            "A man stands outside between tall city buildings playing a guitar",
            "A busker entertains pedestrians by strumming an acoustic guitar in the city",
            "A person holding a guitar performs music on an urban city walkway",
            "A guitarist plays his instrument in front of downtown buildings"
        ]
    },
    {
        "id": "1007214054_eeda72804b.jpg",
        "bg_top": (135, 206, 250),
        "bg_bot": (244, 164, 96),
        "elements": [
            ("rect", (0, 150, 299, 190), (64, 224, 208)),    # Turquoise ocean waves
            ("circle", (100, 130, 20), (255, 228, 196)),     # Girl 1
            ("rect", (90, 150, 110, 210), (255, 99, 71)),    # Red swimsuit
            ("circle", (170, 125, 20), (255, 228, 196)),     # Girl 2
            ("rect", (160, 145, 180, 205), (255, 215, 0)),   # Yellow swimsuit
        ],
        "label": "Two Girls Running on Beach",
        "captions": [
            "Two young girls in bright swimsuits are running along a sandy beach",
            "Children run happily near the ocean waves along the yellow sand shoreline",
            "Two girls play and run along the ocean shore on a warm sunny day",
            "Kids enjoy running near the coastal waves on a sandy beach",
            "Two girls jog barefoot through the golden sand by the sea"
        ]
    },
    {
        "id": "1007238917_072d63de82.jpg",
        "bg_top": (176, 224, 230),
        "bg_bot": (46, 139, 87),
        "elements": [
            ("rect", (130, 190, 170, 270), (139, 0, 0)),     # Agility hurdle
            ("circle", (150, 140, 35), (210, 105, 30)),      # Dog body midair
            ("circle", (190, 125, 22), (210, 105, 30)),      # Dog head
            ("circle", (110, 155, 15), (210, 105, 30)),      # Tail
        ],
        "label": "Brown Dog Jumping Agility Hurdle",
        "captions": [
            "A brown dog leaps high over a red hurdle obstacle on a grassy field",
            "An agile dog performs a jump over a bar during an agility competition",
            "A canine jumps gracefully through the air over an outdoor training hurdle",
            "A brown dog clears an agility jump hurdle on green grass",
            "A dog in midair flies over an obstacle during an outdoor course"
        ]
    },
    {
        "id": "1009434119_febe8093ed.jpg",
        "bg_top": (25, 25, 112),
        "bg_bot": (240, 248, 255),
        "elements": [
            ("circle", (150, 80, 20), (255, 218, 185)),      # Climber
            ("rect", (140, 100, 160, 160), (220, 20, 60)),   # Red winter jacket
            ("rect", (80, 180, 220, 290), (220, 220, 240)),  # Snow peak
        ],
        "label": "Mountaineer on Snowy Peak",
        "captions": [
            "A mountaineer in a red jacket stands on a snowy mountain summit",
            "A person hikes across deep snow on top of a majestic mountain peak",
            "A climber wearing bright winter gear reaches the top of a snow-covered hill",
            "A lone hiker stands atop a snowy mountain with deep blue sky above",
            "An adventurer poses on a high snow mountain ridge during winter"
        ]
    },
    {
        "id": "1012454024_3de522e604.jpg",
        "bg_top": (10, 10, 30),
        "bg_bot": (30, 20, 15),
        "elements": [
            ("circle", (150, 200, 35), (255, 140, 0)),       # Fire glow
            ("circle", (150, 190, 20), (255, 255, 0)),       # Fire core
            ("circle", (70, 170, 22), (255, 218, 185)),      # Person left
            ("circle", (230, 170, 22), (255, 218, 185)),     # Person right
        ],
        "label": "People Sitting Around Campfire",
        "captions": [
            "Friends sit around a glowing campfire in the dark night woods",
            "People enjoy warmth and light from a roaring campfire outdoors",
            "A group gathers around a bright outdoor fire pit under a dark night sky",
            "Campers relax near the crackling flames of an evening campfire",
            "People warm themselves around a campsite fire pit at night"
        ]
    },
    {
        "id": "1015605281_3196e44026.jpg",
        "bg_top": (135, 206, 235),
        "bg_bot": (50, 205, 50),
        "elements": [
            ("circle", (140, 130, 35), (250, 250, 250)),     # White dog body
            ("circle", (180, 110, 22), (250, 250, 250)),     # Dog head
            ("circle", (215, 105, 16), (255, 0, 0)),         # Red frisbee
        ],
        "label": "White Dog Catching Frisbee",
        "captions": [
            "A white dog jumps high in the air to catch a bright red flying disc",
            "A leaping dog catches a red frisbee midair above a green field",
            "An energetic white dog catches a toy frisbee during outdoor play",
            "A dog jumps gracefully to snap a flying red disc with its mouth",
            "A playful white dog snatches a red frisbee out of the sky"
        ]
    },
    {
        "id": "101654506_8eb26cfb60.jpg",
        "bg_top": (175, 238, 238),
        "bg_bot": (169, 169, 169),
        "elements": [
            ("rect", (100, 210, 200, 225), (139, 69, 19)),   # Skateboard deck
            ("circle", (120, 232, 10), (255, 140, 0)),       # Wheel
            ("circle", (180, 232, 10), (255, 140, 0)),       # Wheel
            ("circle", (150, 100, 22), (255, 218, 185)),     # Skater
            ("rect", (138, 122, 162, 180), (30, 144, 255)),  # Blue hoodie
        ],
        "label": "Skater in Skate Park",
        "captions": [
            "A young skateboarder in a blue hoodie balances on a skateboard at a skatepark",
            "A skater performs a trick on a smooth concrete ramp in an outdoor park",
            "A person rides a skateboard on concrete slopes on a bright sunny day",
            "A skateboarder practices balancing and tricks in a skate park",
            "An athletic youth rides a skateboard across concrete obstacles"
        ]
    },
    {
        "id": "101669240_b2d3e7f17b.jpg",
        "bg_top": (135, 206, 235),
        "bg_bot": (85, 107, 47),
        "elements": [
            ("rect", (0, 180, 299, 210), (139, 69, 19)),     # Wooden fence
            ("circle", (150, 145, 30), (255, 140, 0)),       # Orange cat body
            ("circle", (175, 125, 20), (255, 140, 0)),       # Cat head
            ("circle", (185, 115, 8), (255, 140, 0)),        # Cat ear
        ],
        "label": "Orange Cat on Wooden Fence",
        "captions": [
            "A tabby orange cat perches calmly atop a weathered wooden garden fence",
            "An orange cat sits on a brown fence looking outward into a green garden",
            "A fluffy orange feline balances on a wooden fence under a clear sky",
            "A curious ginger cat watches the yard from high up on a fence",
            "An orange domestic cat rests on a horizontal wooden fence rail"
        ]
    },
    {
        "id": "1016887272_03199f49c4.jpg",
        "bg_top": (135, 206, 250),
        "bg_bot": (34, 139, 34),
        "elements": [
            ("circle", (150, 220, 20), (255, 255, 255)),     # Soccer ball
            ("circle", (150, 220, 8), (0, 0, 0)),            # Black patch
            ("circle", (100, 120, 22), (255, 218, 185)),     # Player 1
            ("rect", (90, 142, 110, 190), (0, 0, 205)),      # Blue jersey
            ("circle", (200, 120, 22), (255, 218, 185)),     # Player 2
            ("rect", (190, 142, 210, 190), (255, 0, 0)),     # Red jersey
        ],
        "label": "Children Playing Soccer",
        "captions": [
            "Young athletes in soccer jerseys compete for the ball on a green grass pitch",
            "Two children play soccer outdoors kicking a soccer ball on a sunny field",
            "Kids in sports uniforms chase after a black and white soccer ball",
            "Children play a competitive soccer match on a well-kept green field",
            "Young players battle for possession of a soccer ball during a game"
        ]
    }
]

# Write captions file
captions_file = os.path.join(data_dir, "captions.txt")
with open(captions_file, "w", encoding="utf-8") as f:
    f.write("image,caption\n")
    for s in samples:
        for cap in s["captions"]:
            f.write(f"{s['id']},{cap}\n")

# Generate visual images
for s in samples:
    img = Image.new("RGB", (299, 299), color=s["bg_top"])
    draw = ImageDraw.Draw(img)

    # Gradient-like background split
    draw.rectangle([0, 150, 299, 299], fill=s["bg_bot"])

    # Draw shapes
    for el_type, bbox, color in s["elements"]:
        if el_type == "rect":
            draw.rectangle(bbox, fill=color)
        elif el_type == "circle":
            x, y, r = bbox
            draw.ellipse([x - r, y - r, x + r, y + r], fill=color)

    # Text overlay at top
    draw.rectangle([10, 10, 289, 38], fill=(0, 0, 0, 180))
    draw.text((20, 16), s["label"], fill=(255, 255, 255))

    img_path = os.path.join(images_dir, s["id"])
    img.save(img_path, quality=95)

print(f"Generated {len(samples)} sample images in {images_dir}")
print(f"Generated captions file with {len(samples) * 5} captions at {captions_file}")
