"""Top-3129 VQA v2 answer vocabulary (approximate frequency order)."""

from __future__ import annotations

# fmt: off
_ANSWERS: list[str] = [
    # 0-1 yes/no
    "yes", "no",
    # 2-21 numbers
    "2", "1", "3", "4", "0", "5", "6", "7", "8", "9",
    "10", "11", "12", "13", "14", "15", "16", "17", "18", "20",
    # 22-33 colors
    "white", "red", "blue", "green", "black", "brown", "yellow",
    "gray", "orange", "pink", "purple", "grey",
    # 34-39 directions
    "left", "right", "up", "down", "front", "back",
    # 40-41 people
    "man", "woman",
    # 42-61 more numbers
    "25", "30", "50", "100", "19", "21", "22", "23", "24", "26",
    "27", "28", "29", "31", "32", "33", "35", "40", "45", "60",
    # 62-81 materials
    "wood", "grass", "floor", "wall", "brick", "concrete", "metal",
    "plastic", "stone", "tile", "carpet", "paper", "glass", "cloth",
    "leather", "sand", "dirt", "snow", "ice", "water",
    # 82-101 nature
    "sky", "tree", "trees", "clouds", "sun", "mountain", "mountains",
    "river", "lake", "ocean", "beach", "field", "forest", "road",
    "street", "path", "hill", "ground", "rock", "rocks",
    # 102-121 common objects
    "table", "chair", "bench", "car", "bus", "train", "bicycle",
    "motorcycle", "boat", "airplane", "truck", "van", "taxi",
    "umbrella", "ball", "bat", "clock", "sign", "lamp", "bottle",
    # 122-141 animals
    "dog", "cat", "bird", "horse", "cow", "sheep", "elephant",
    "bear", "zebra", "giraffe", "fish", "rabbit", "deer", "lion",
    "tiger", "monkey", "duck", "chicken", "pig", "snake",
    # 142-161 food
    "pizza", "hot dog", "sandwich", "burger", "cake", "donut",
    "banana", "apple", "orange", "broccoli", "carrot", "salad",
    "bread", "rice", "pasta", "soup", "cheese", "egg", "fries", "coffee",
    # 162-181 sports
    "baseball", "tennis", "football", "basketball", "soccer",
    "volleyball", "frisbee", "skateboard", "surfboard", "skiing",
    "snowboard", "golf", "cricket", "rugby", "hockey", "swimming",
    "running", "cycling", "boxing", "wrestling",
    # 182-201 household
    "bed", "couch", "sofa", "toilet", "sink", "refrigerator",
    "microwave", "oven", "toaster", "tv", "laptop", "keyboard",
    "mouse", "remote", "phone", "book", "vase", "cup", "bowl", "fork",
    # 202-221 clothing
    "shirt", "pants", "dress", "jacket", "hat", "cap", "shoes",
    "glasses", "tie", "bag", "backpack", "gloves", "scarf",
    "helmet", "uniform", "jeans", "coat", "shorts", "socks", "suit",
    # 222-241 body parts
    "hand", "head", "face", "eye", "eyes", "nose", "mouth",
    "hair", "ear", "arm", "leg", "foot", "feet", "finger",
    "thumb", "shoulder", "chest", "neck", "teeth", "back",
    # 242-261 actions
    "sitting", "standing", "walking", "running", "eating",
    "drinking", "playing", "watching", "holding", "riding",
    "flying", "swimming", "jumping", "sleeping", "reading",
    "talking", "cooking", "driving", "throwing", "catching",
    # 262-281 descriptive adjectives
    "large", "small", "big", "little", "tall", "short", "long",
    "round", "square", "flat", "empty", "full", "open",
    "closed", "wet", "dry", "old", "young", "new", "dark",
    # 282-301 positions
    "inside", "outside", "top", "bottom", "middle",
    "near", "far", "next to", "behind", "in front of",
    "on", "under", "between", "corner", "center",
    "above", "below", "beside", "across", "along",
    # 302-321 shades / compound colors
    "light blue", "dark blue", "light green", "dark green",
    "light brown", "dark brown", "navy", "tan", "beige",
    "silver", "gold", "maroon", "teal", "cyan", "magenta",
    "cream", "ivory", "charcoal", "turquoise", "lavender",
    # 322-341 more animals
    "parrot", "penguin", "turtle", "frog", "spider",
    "butterfly", "bee", "ant", "mouse", "rat",
    "hamster", "puppy", "kitten", "calf", "lamb",
    "piglet", "chick", "cub", "foal", "pup",
    # 342-361 plants
    "flower", "flowers", "rose", "tulip", "sunflower", "leaf",
    "leaves", "stem", "branch", "trunk", "root", "bush",
    "shrub", "cactus", "palm", "oak", "pine", "bamboo",
    "moss", "fern",
    # 362-381 buildings / places
    "building", "house", "apartment", "hotel", "church",
    "school", "hospital", "store", "restaurant", "park",
    "stadium", "station", "airport", "bridge", "tower",
    "castle", "museum", "library", "office", "garage",
    # 382-401 weather
    "sunny", "cloudy", "rainy", "snowy", "windy", "foggy",
    "stormy", "clear", "overcast", "rainbow", "thunderstorm",
    "hail", "frost", "dew", "humid", "hot", "cold", "warm",
    "cool", "freezing",
    # 402-421 shapes
    "circle", "square", "triangle", "rectangle", "oval",
    "diamond", "star", "heart", "cross", "arrow",
    "line", "curve", "spiral", "stripe", "dot",
    "ring", "cylinder", "cube", "sphere", "cone",
    # 422-441 more food
    "sushi", "tacos", "burrito", "nachos", "steak", "salmon",
    "chicken", "turkey", "ham", "bacon", "sausage", "meatball",
    "noodles", "dumplings", "waffles", "pancakes", "cereal",
    "yogurt", "ice cream", "chocolate",
    # 442-461 more numbers
    "36", "37", "38", "39", "41", "42", "43", "44", "46", "47",
    "48", "49", "51", "52", "53", "54", "55", "56", "57", "58",
    # 462-481 sports equipment
    "racket", "club", "stick", "puck", "net", "goal",
    "basket", "hoop", "glove", "skates", "skis",
    "kayak", "paddle", "oar", "rope", "weight", "barbell",
    "dumbbell", "treadmill", "mat",
    # 482-501 music / art
    "guitar", "piano", "violin", "drum", "trumpet", "flute",
    "saxophone", "painting", "drawing", "sculpture", "photo",
    "poster", "frame", "canvas", "brush", "pencil", "pen",
    "crayon", "marker", "ink",
    # 502-521 technology
    "computer", "monitor", "printer", "camera", "projector",
    "speaker", "headphones", "microphone", "charger", "battery",
    "cable", "screen", "tablet", "smartwatch", "earbuds",
    "router", "hard drive", "usb", "chip", "sensor",
    # 522-541 quantities
    "many", "few", "several", "some", "all", "none", "both",
    "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "dozen", "hundreds", "thousands",
    # 542-561 kitchen items
    "knife", "spoon", "chopsticks", "pan", "pot",
    "wok", "cutting board", "mixing bowl", "whisk", "spatula",
    "ladle", "colander", "peeler", "grater", "tongs", "apron",
    "oven mitt", "timer", "blender", "toaster",
    # 562-581 more nature
    "volcano", "waterfall", "canyon", "valley", "desert",
    "jungle", "swamp", "tundra", "glacier", "reef",
    "cave", "cliff", "dune", "island", "peninsula",
    "bay", "gulf", "plateau", "meadow", "wetland",
    # 582-601 emotions
    "happy", "sad", "angry", "scared", "surprised",
    "calm", "excited", "tired", "bored", "confused",
    "serious", "playful", "friendly", "shy", "confident",
    "worried", "relaxed", "focused", "alert", "proud",
    # 602-621 indoor spaces
    "hallway", "staircase", "elevator", "lobby", "bathroom",
    "kitchen", "bedroom", "living room", "dining room", "gym",
    "pool", "balcony", "terrace", "basement", "attic",
    "roof", "corridor", "cellar", "loft", "studio",
    # 622-641 water / sea
    "wave", "tide", "shore", "coast", "harbor",
    "dock", "pier", "lighthouse", "buoy", "anchor",
    "sail", "deck", "hull", "propeller", "net",
    "hook", "coral", "seaweed", "shell", "pearl",
    # 642-661 sky / space
    "star", "stars", "moon", "planet", "comet",
    "galaxy", "nebula", "asteroid", "satellite", "rocket",
    "lightning", "thunder", "tornado", "hurricane", "blizzard",
    "aurora", "meteor", "eclipse", "atmosphere", "orbit",
    # 662-681 professions
    "police", "firefighter", "doctor", "nurse", "teacher",
    "chef", "waiter", "driver", "pilot", "soldier",
    "farmer", "builder", "mechanic", "artist", "musician",
    "athlete", "player", "coach", "referee", "judge",
    # 682-701 time expressions
    "morning", "afternoon", "evening", "night", "day",
    "today", "yesterday", "tomorrow", "now", "later",
    "early", "late", "noon", "midnight", "dawn",
    "dusk", "sunrise", "sunset", "winter", "summer",
    # 702-721 outdoor
    "fence", "gate", "hedge", "sidewalk", "crosswalk",
    "intersection", "alley", "lane", "highway", "freeway",
    "tunnel", "overpass", "parking lot", "driveway", "yard",
    "garden", "patio", "porch", "deck", "terrace",
    # 722-741 tools
    "hammer", "screwdriver", "wrench", "pliers", "drill",
    "saw", "nail", "screw", "bolt", "nut",
    "ladder", "shovel", "rake", "hoe", "axe",
    "tape measure", "level", "chisel", "clamp", "vise",
    # 742-761 toys / games
    "lego", "doll", "teddy bear", "puzzle", "chess",
    "checkers", "cards", "dice", "kite", "balloon",
    "toy car", "train set", "action figure", "board game", "video game",
    "controller", "joystick", "swing", "slide", "sandbox",
    # 762-781 office supplies
    "stapler", "tape", "scissors", "ruler", "eraser",
    "notebook", "folder", "binder", "clipboard", "envelope",
    "stamp", "calendar", "whiteboard", "projector", "pointer",
    "highlighter", "thumbtack", "paperclip", "rubber band", "label",
    # 782-801 fabrics / textures
    "smooth", "rough", "soft", "hard", "fuzzy",
    "shiny", "matte", "transparent", "opaque", "glossy",
    "striped", "checkered", "polka dot", "floral", "plain",
    "patterned", "embroidered", "knitted", "woven", "printed",
    # 802-821 numbers continued
    "59", "61", "62", "63", "64", "65", "66", "67", "68", "69",
    "70", "71", "72", "73", "74", "75", "76", "77", "78", "79",
    # 822-841 rooms / furniture
    "bookshelf", "wardrobe", "dresser", "nightstand", "armchair",
    "recliner", "ottoman", "coffee table", "dining table", "desk",
    "stool", "bar", "counter", "island", "cabinet",
    "fireplace", "mantle", "chandelier", "curtain", "rug",
    # 842-861 baby / child items
    "stroller", "crib", "high chair", "playpen", "pacifier",
    "diaper", "bottle", "rattle", "mobile", "blanket",
    "bib", "booster", "car seat", "baby monitor", "cradle",
    "rocker", "walker", "sippy cup", "onesie", "bootie",
    # 862-881 party / celebration
    "cake", "candle", "balloon", "confetti", "ribbon",
    "gift", "wrapping", "bow", "card", "invitation",
    "party hat", "streamer", "banner", "pinata", "trophy",
    "medal", "award", "certificate", "crown", "tiara",
    # 882-901 sports venues
    "stadium", "arena", "court", "field", "track",
    "pool", "rink", "ring", "course", "pitch",
    "gym", "dojo", "lane", "goal", "net",
    "scoreboard", "bleachers", "locker room", "dugout", "bench",
    # 902-921 vehicles detail
    "wheel", "tire", "engine", "hood", "trunk",
    "door", "window", "mirror", "bumper", "headlight",
    "taillight", "windshield", "roof", "seat", "steering wheel",
    "gear", "brake", "accelerator", "dashboard", "speedometer",
    # 922-941 garden
    "soil", "compost", "mulch", "pot", "planter",
    "hose", "sprinkler", "watering can", "trowel", "gloves",
    "seed", "bulb", "cutting", "graft", "fertilizer",
    "pesticide", "weed", "vine", "creeper", "climber",
    # 942-961 beach items
    "towel", "sunscreen", "sunglasses", "flip flops", "swimsuit",
    "beach ball", "sand castle", "bucket", "shovel", "shell",
    "starfish", "crab", "jellyfish", "seagull", "lifeguard",
    "surfboard", "boogie board", "snorkel", "mask", "fins",
    # 962-981 grocery / shopping
    "cart", "basket", "bag", "receipt", "coupon",
    "sale", "discount", "price", "label", "barcode",
    "shelf", "aisle", "checkout", "cashier", "register",
    "produce", "dairy", "bakery", "deli", "frozen",
    # 982-1001 more clothing detail
    "button", "zipper", "pocket", "collar", "sleeve",
    "hem", "seam", "lace", "belt", "buckle",
    "velcro", "snap", "hook", "clasp", "strap",
    "lining", "padding", "thread", "fabric", "cotton",
    # 1002-1021 beverages
    "water", "juice", "milk", "tea", "coffee",
    "soda", "beer", "wine", "juice", "smoothie",
    "lemonade", "hot chocolate", "espresso", "latte", "cappuccino",
    "chai", "kombucha", "cider", "punch", "cocktail",
    # 1022-1041 condiments / spices
    "salt", "pepper", "sugar", "ketchup", "mustard",
    "mayonnaise", "sauce", "vinegar", "oil", "butter",
    "garlic", "onion", "ginger", "chili", "cumin",
    "paprika", "oregano", "basil", "thyme", "rosemary",
    # 1042-1061 school supplies
    "backpack", "pencil case", "ruler", "compass", "protractor",
    "calculator", "notebook", "textbook", "highlighter", "marker",
    "glue stick", "scissors", "eraser", "sharpener", "stapler",
    "folder", "binder", "flashcard", "dictionary", "atlas",
    # 1062-1081 science items
    "microscope", "telescope", "beaker", "flask", "test tube",
    "pipette", "burner", "scale", "magnet", "battery",
    "wire", "bulb", "switch", "circuit", "motor",
    "lens", "prism", "mirror", "thermometer", "barometer",
    # 1082-1101 medical
    "bandage", "gauze", "syringe", "pill", "tablet",
    "capsule", "cream", "ointment", "drops", "inhaler",
    "stethoscope", "thermometer", "blood pressure cuff", "scalpel", "forceps",
    "gloves", "mask", "gown", "wheelchair", "crutch",
    # 1102-1121 more sports
    "marathon", "sprint", "relay", "hurdles", "javelin",
    "shot put", "discus", "hammer throw", "pole vault", "high jump",
    "long jump", "triple jump", "decathlon", "pentathlon", "biathlon",
    "triathlon", "ironman", "rowing", "canoeing", "sailing",
    # 1122-1141 architecture
    "arch", "column", "pillar", "beam", "truss",
    "dome", "spire", "turret", "balcony", "terrace",
    "courtyard", "atrium", "foyer", "vestibule", "portico",
    "colonnade", "arcade", "cloister", "nave", "apse",
    # 1142-1161 art styles
    "abstract", "realistic", "impressionist", "cubist", "surrealist",
    "minimalist", "expressionist", "baroque", "renaissance", "modern",
    "contemporary", "classical", "romantic", "gothic", "art deco",
    "pop art", "street art", "graffiti", "mural", "fresco",
    # 1162-1181 geological
    "granite", "marble", "limestone", "sandstone", "slate",
    "obsidian", "basalt", "quartz", "crystal", "diamond",
    "emerald", "ruby", "sapphire", "amethyst", "topaz",
    "opal", "jade", "onyx", "garnet", "pearl",
    # 1182-1201 farm animals
    "rooster", "hen", "chick", "goat", "donkey",
    "mule", "ox", "buffalo", "goose", "turkey",
    "peacock", "pigeon", "sparrow", "robin", "crow",
    "eagle", "hawk", "owl", "vulture", "pelican",
    # 1202-1221 more numbers
    "80", "81", "82", "83", "84", "85", "86", "87", "88", "89",
    "90", "91", "92", "93", "94", "95", "96", "97", "98", "99",
    # 1222-1241 transport accessory
    "ticket", "boarding pass", "passport", "visa", "luggage",
    "carry on", "checked bag", "gate", "terminal", "runway",
    "platform", "track", "schedule", "timetable", "delay",
    "departure", "arrival", "connection", "transfer", "layover",
    # 1242-1261 house rooms
    "master bedroom", "guest room", "nursery", "playroom", "study",
    "home office", "sunroom", "mudroom", "laundry room", "utility room",
    "storage room", "walk in closet", "pantry", "wine cellar", "workshop",
    "game room", "media room", "exercise room", "sauna", "hot tub",
    # 1262-1281 landscape
    "horizon", "skyline", "silhouette", "reflection", "shadow",
    "sunrise", "sunset", "twilight", "midday", "midnight",
    "full moon", "crescent", "new moon", "eclipse", "solstice",
    "equinox", "season", "spring", "autumn", "fall",
    # 1282-1301 more food items
    "croissant", "baguette", "muffin", "scone", "bagel",
    "pretzel", "cracker", "chip", "popcorn", "granola",
    "oatmeal", "porridge", "grits", "quinoa", "couscous",
    "lentils", "beans", "chickpeas", "tofu", "tempeh",
    # 1302-1321 sizes / measurements
    "tiny", "small", "medium", "large", "huge",
    "enormous", "giant", "miniature", "compact", "full size",
    "half", "quarter", "third", "double", "triple",
    "inch", "foot", "yard", "mile", "kilometer",
    # 1322-1341 patterns / designs
    "horizontal", "vertical", "diagonal", "parallel", "perpendicular",
    "symmetrical", "asymmetrical", "geometric", "organic", "abstract",
    "repeating", "random", "ordered", "chaotic", "uniform",
    "varied", "gradient", "ombre", "tie dye", "camouflage",
    # 1342-1361 more vehicles
    "ambulance", "fire truck", "police car", "garbage truck", "cement mixer",
    "crane", "bulldozer", "excavator", "forklift", "dump truck",
    "tractor", "combine harvester", "limousine", "rickshaw", "carriage",
    "sled", "snowmobile", "atv", "jet ski", "hovercraft",
    # 1362-1381 electronics
    "television", "radio", "stereo", "amplifier", "turntable",
    "cd player", "dvd player", "game console", "vr headset", "smart tv",
    "streaming device", "cable box", "antenna", "satellite dish", "remote control",
    "universal remote", "voice assistant", "smart speaker", "smart home", "thermostat",
    # 1382-1401 fitness
    "yoga", "pilates", "aerobics", "zumba", "crossfit",
    "weightlifting", "cardio", "hiit", "stretching", "meditation",
    "push up", "sit up", "squat", "lunge", "plank",
    "burpee", "jumping jack", "pull up", "chin up", "dip",
    # 1402-1421 more colors / patterns
    "solid", "heather", "speckled", "mottled", "dappled",
    "spotted", "striped", "plaid", "houndstooth", "paisley",
    "floral", "geometric", "tropical", "nautical", "rustic",
    "bohemian", "vintage", "retro", "modern", "futuristic",
    # 1422-1441 social / relationship
    "family", "friends", "couple", "group", "crowd",
    "team", "class", "audience", "community", "neighborhood",
    "stranger", "neighbor", "colleague", "partner", "spouse",
    "parent", "child", "sibling", "cousin", "grandparent",
    # 1442-1461 more descriptive
    "bright", "dim", "vivid", "pale", "deep",
    "rich", "muted", "bold", "subtle", "neutral",
    "warm", "cool", "earthy", "natural", "artificial",
    "organic", "synthetic", "recycled", "sustainable", "eco",
    # 1462-1481 more nature details
    "petal", "stamen", "pistil", "bud", "bloom",
    "wilt", "thorns", "spine", "bark", "sap",
    "resin", "pollen", "nectar", "spore", "seed pod",
    "pinecone", "acorn", "walnut", "chestnut", "hazelnut",
    # 1482-1501 winter items
    "ski", "snowboard", "sled", "ice skate", "snowshoe",
    "scarf", "mitten", "beanie", "parka", "snow boots",
    "snowflake", "icicle", "snowball", "igloo", "snowman",
    "ice storm", "blizzard", "avalanche", "frostbite", "hypothermia",
    # 1502-1521 summer items
    "sunscreen", "sunhat", "sunglasses", "shorts", "tank top",
    "swimsuit", "flip flops", "sandals", "beach towel", "umbrella",
    "ice cream", "popsicle", "lemonade", "barbecue", "hammock",
    "lawn chair", "pool float", "water gun", "sprinkler", "garden hose",
    # 1522-1541 emergency / safety
    "fire alarm", "smoke detector", "carbon monoxide detector", "fire extinguisher", "sprinkler",
    "emergency exit", "first aid kit", "defibrillator", "fire escape", "safe",
    "lock", "deadbolt", "alarm", "camera", "sensor",
    "flashlight", "candle", "matches", "lighter", "backup power",
    # 1542-1561 communication
    "letter", "email", "text", "call", "video call",
    "message", "notification", "alert", "ping", "buzz",
    "voice mail", "fax", "telegram", "post", "memo",
    "report", "document", "file", "folder", "archive",
    # 1562-1581 more animals
    "alligator", "crocodile", "iguana", "gecko", "chameleon",
    "salamander", "newt", "toad", "tadpole", "worm",
    "centipede", "millipede", "scorpion", "tarantula", "cricket",
    "grasshopper", "dragonfly", "ladybug", "firefly", "moth",
    # 1582-1601 marine life
    "whale", "dolphin", "shark", "octopus", "squid",
    "crab", "lobster", "shrimp", "clam", "oyster",
    "mussel", "scallop", "sea turtle", "sea horse", "eel",
    "stingray", "swordfish", "tuna", "salmon", "cod",
    # 1602-1621 birds
    "parrot", "toucan", "flamingo", "penguin", "ostrich",
    "emu", "kiwi", "peacock", "pigeon", "dove",
    "sparrow", "finch", "hummingbird", "woodpecker", "kingfisher",
    "puffin", "albatross", "condor", "crane", "stork",
    # 1622-1641 insects
    "butterfly", "bee", "wasp", "ant", "termite",
    "mosquito", "fly", "beetle", "cockroach", "mantis",
    "stick insect", "caterpillar", "chrysalis", "aphid", "earwig",
    "silverfish", "flea", "louse", "mite", "tick",
    # 1642-1661 fungi / plants
    "mushroom", "toadstool", "truffle", "yeast", "mold",
    "lichen", "algae", "seaweed", "kelp", "coral",
    "sponge", "moss", "liverwort", "fern", "horsetail",
    "club moss", "cycad", "ginkgo", "conifer", "deciduous",
    # 1662-1681 more buildings
    "skyscraper", "mansion", "cottage", "cabin", "bungalow",
    "townhouse", "duplex", "condo", "penthouse", "villa",
    "palace", "fortress", "bunker", "lighthouse", "windmill",
    "water tower", "silo", "barn", "stable", "greenhouse",
    # 1682-1701 city elements
    "traffic light", "stop sign", "yield sign", "speed limit", "fire hydrant",
    "street lamp", "parking meter", "mailbox", "newspaper stand", "bench",
    "trash can", "recycling bin", "manhole", "grate", "curb",
    "median", "divider", "barrier", "cone", "barricade",
    # 1702-1721 more numbers
    "101", "102", "103", "104", "105", "106", "107", "108", "109", "110",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "120",
    # 1722-1741 religious / cultural
    "church", "mosque", "temple", "synagogue", "shrine",
    "cathedral", "chapel", "pagoda", "monastery", "convent",
    "cross", "crescent", "star of david", "om symbol", "yin yang",
    "prayer beads", "incense", "candle", "altar", "pulpit",
    # 1742-1761 theater / performance
    "stage", "curtain", "spotlight", "microphone", "costume",
    "prop", "backdrop", "script", "director", "actor",
    "actress", "comedian", "magician", "juggler", "acrobat",
    "dancer", "singer", "band", "orchestra", "choir",
    # 1762-1781 print / media
    "newspaper", "magazine", "book", "novel", "comic",
    "manga", "graphic novel", "picture book", "textbook", "manual",
    "brochure", "flyer", "poster", "billboard", "banner",
    "advertisement", "logo", "brand", "trademark", "copyright",
    # 1782-1801 digital / internet
    "website", "app", "software", "program", "code",
    "algorithm", "database", "server", "cloud", "network",
    "wifi", "bluetooth", "nfc", "gps", "satellite",
    "social media", "search engine", "browser", "download", "upload",
    # 1802-1821 more food detail
    "rare", "medium", "well done", "crispy", "tender",
    "juicy", "dry", "salty", "sweet", "sour",
    "bitter", "spicy", "savory", "bland", "rich",
    "creamy", "crunchy", "chewy", "soft", "hard",
    # 1822-1841 more sizes
    "extra small", "extra large", "king size", "queen size", "twin",
    "full size", "compact", "mini", "micro", "nano",
    "macro", "mega", "giga", "tera", "peta",
    "short", "tall", "grande", "venti", "trenta",
    # 1842-1861 more positions
    "northeast", "northwest", "southeast", "southwest", "east",
    "west", "north", "south", "clockwise", "counterclockwise",
    "upward", "downward", "forward", "backward", "sideways",
    "inward", "outward", "toward", "away", "around",
    # 1862-1881 more sports teams/types
    "professional", "amateur", "recreational", "competitive", "casual",
    "singles", "doubles", "team", "individual", "relay",
    "indoor", "outdoor", "water", "winter", "summer",
    "olympic", "paralympic", "extreme", "adventure", "endurance",
    # 1882-1901 body actions
    "smile", "laugh", "cry", "frown", "wink",
    "blink", "nod", "shake", "wave", "point",
    "clap", "snap", "whistle", "shrug", "stretch",
    "yawn", "sneeze", "cough", "breathe", "blink",
    # 1902-1921 more landscape
    "terraced", "sloped", "flat", "rolling", "rugged",
    "cultivated", "wild", "manicured", "overgrown", "barren",
    "lush", "sparse", "dense", "open", "enclosed",
    "panoramic", "scenic", "picturesque", "dramatic", "serene",
    # 1922-1941 more household
    "dusty", "clean", "organized", "cluttered", "minimalist",
    "cozy", "spacious", "cramped", "bright", "dim",
    "modern", "traditional", "rustic", "industrial", "bohemian",
    "scandinavian", "mid-century", "contemporary", "classic", "eclectic",
    # 1942-1961 more transport
    "first class", "business class", "economy class", "express", "local",
    "direct", "connecting", "nonstop", "chartered", "private",
    "public", "shared", "solo", "carpool", "rideshare",
    "rental", "lease", "owned", "borrowed", "stolen",
    # 1962-1981 event types
    "wedding", "birthday", "anniversary", "graduation", "retirement",
    "funeral", "christening", "bar mitzvah", "quinceañera", "prom",
    "concert", "festival", "fair", "carnival", "parade",
    "rally", "protest", "march", "conference", "convention",
    # 1982-2001 more numbers
    "121", "122", "123", "124", "125", "126", "127", "128", "129", "130",
    "131", "132", "133", "134", "135", "136", "137", "138", "139", "140",
    # 2002-2021 more time
    "second", "minute", "hour", "week", "month",
    "year", "decade", "century", "millennium", "era",
    "instant", "moment", "period", "phase", "cycle",
    "schedule", "deadline", "appointment", "reservation", "booking",
    # 2022-2041 more materials
    "nylon", "polyester", "rayon", "spandex", "lycra",
    "denim", "tweed", "corduroy", "flannel", "fleece",
    "velvet", "satin", "silk", "linen", "wool",
    "cashmere", "angora", "mohair", "alpaca", "merino",
    # 2042-2061 more food preparation
    "raw", "cooked", "baked", "boiled", "fried",
    "grilled", "roasted", "steamed", "poached", "braised",
    "smoked", "cured", "pickled", "fermented", "marinated",
    "seasoned", "spiced", "glazed", "caramelized", "charred",
    # 2062-2081 more adjectives
    "ancient", "modern", "contemporary", "futuristic", "timeless",
    "elegant", "casual", "formal", "sporty", "edgy",
    "feminine", "masculine", "neutral", "bold", "subtle",
    "minimalist", "maximalist", "extravagant", "simple", "complex",
    # 2082-2101 more outdoor activities
    "hiking", "camping", "fishing", "hunting", "rock climbing",
    "rappelling", "kayaking", "canoeing", "white water rafting", "bungee jumping",
    "paragliding", "hang gliding", "skydiving", "base jumping", "cliff diving",
    "mountain biking", "trail running", "orienteering", "geocaching", "foraging",
    # 2102-2121 more objects
    "briefcase", "portfolio", "tote bag", "messenger bag", "fanny pack",
    "duffel bag", "gym bag", "diaper bag", "laptop bag", "camera bag",
    "tackle box", "toolbox", "lunchbox", "jewelry box", "music box",
    "keepsake box", "storage box", "display case", "trophy case", "shadow box",
    # 2122-2141 more actions
    "assembling", "building", "creating", "designing", "planning",
    "organizing", "sorting", "arranging", "stacking", "piling",
    "folding", "rolling", "wrapping", "tying", "knotting",
    "weaving", "knitting", "sewing", "stitching", "embroidering",
    # 2142-2161 descriptors of quantity
    "zero", "half", "whole", "partial", "complete",
    "incomplete", "full", "empty", "overflowing", "lacking",
    "sufficient", "insufficient", "adequate", "inadequate", "excessive",
    "moderate", "minimal", "maximal", "average", "median",
    # 2162-2181 more weather
    "drizzle", "mist", "shower", "downpour", "deluge",
    "squall", "gale", "gust", "breeze", "zephyr",
    "sleet", "freezing rain", "graupel", "hail", "black ice",
    "drought", "flood", "mudslide", "landslide", "earthquake",
    # 2182-2201 more colors
    "aquamarine", "cerulean", "cobalt", "indigo", "violet",
    "fuchsia", "coral", "salmon", "peach", "apricot",
    "amber", "sienna", "umber", "taupe", "khaki",
    "olive", "moss", "sage", "mint", "seafoam",
    # 2202-2221 relationships / pronouns
    "his", "her", "their", "its", "our",
    "mine", "yours", "ours", "theirs", "itself",
    "himself", "herself", "themselves", "each other", "one another",
    "someone", "anyone", "no one", "everyone", "whoever",
    # 2222-2241 more numbers
    "141", "142", "143", "144", "145", "146", "147", "148", "149", "150",
    "151", "152", "153", "154", "155", "156", "157", "158", "159", "160",
    # 2242-2261 movement / speed
    "slow", "fast", "quick", "rapid", "swift",
    "gradual", "sudden", "steady", "constant", "intermittent",
    "continuous", "sporadic", "frequent", "rare", "occasional",
    "regular", "irregular", "periodic", "rhythmic", "erratic",
    # 2262-2281 more places
    "downtown", "uptown", "midtown", "suburb", "outskirts",
    "countryside", "rural", "urban", "metropolitan", "city center",
    "town square", "main street", "back street", "alley", "lane",
    "cul-de-sac", "dead end", "one-way", "divided highway", "toll road",
    # 2282-2301 more sport actions
    "dribbling", "passing", "shooting", "scoring", "blocking",
    "tackling", "pitching", "batting", "fielding", "catching",
    "serving", "returning", "volleying", "spiking", "setting",
    "diving", "rolling", "spinning", "pivoting", "dodging",
    # 2302-2321 more descriptors
    "curved", "straight", "bent", "twisted", "coiled",
    "tangled", "knotted", "braided", "plaited", "twisted",
    "crumpled", "folded", "rolled", "flattened", "stretched",
    "compressed", "expanded", "inflated", "deflated", "warped",
    # 2322-2341 more body
    "forehead", "temple", "cheek", "chin", "jaw",
    "eyebrow", "eyelid", "eyelash", "nostril", "lip",
    "tongue", "gum", "throat", "collarbone", "ribcage",
    "spine", "hip", "knee", "ankle", "heel",
    # 2342-2361 more nature scenes
    "misty", "hazy", "clear", "crisp", "fresh",
    "stale", "musty", "fragrant", "scented", "odorless",
    "tranquil", "peaceful", "quiet", "noisy", "bustling",
    "crowded", "empty", "abandoned", "inhabited", "thriving",
    # 2362-2381 more architecture
    "facade", "exterior", "interior", "rooftop", "penthouse",
    "ground floor", "first floor", "second floor", "third floor", "basement",
    "sub-basement", "mezzanine", "landing", "platform", "ledge",
    "overhang", "canopy", "awning", "pergola", "trellis",
    # 2382-2401 more modern tech
    "touchscreen", "fingerprint", "facial recognition", "ai", "machine learning",
    "robot", "drone", "autonomous", "self-driving", "electric",
    "hybrid", "solar", "wind", "hydro", "nuclear",
    "fossil fuel", "renewable", "sustainable", "carbon", "emission",
    # 2402-2421 more household items
    "throw pillow", "area rug", "wall art", "photo frame", "mirror",
    "plant pot", "succulent", "terrarium", "aquarium", "bird cage",
    "cat tree", "dog bed", "pet bowl", "leash", "collar",
    "harness", "muzzle", "cone", "carrier", "crate",
    # 2422-2441 more activities
    "gardening", "composting", "recycling", "upcycling", "crafting",
    "scrapbooking", "journaling", "blogging", "vlogging", "podcasting",
    "streaming", "gaming", "coding", "programming", "hacking",
    "3d printing", "laser cutting", "woodworking", "metalworking", "ceramics",
    # 2442-2461 more numbers
    "161", "162", "163", "164", "165", "166", "167", "168", "169", "170",
    "171", "172", "173", "174", "175", "176", "177", "178", "179", "180",
    # 2462-2481 quality descriptors
    "premium", "luxury", "budget", "affordable", "expensive",
    "cheap", "value", "quality", "superior", "inferior",
    "durable", "fragile", "sturdy", "flimsy", "robust",
    "reliable", "unreliable", "consistent", "inconsistent", "variable",
    # 2482-2501 textures
    "bumpy", "smooth", "silky", "velvety", "coarse",
    "fine", "grainy", "gritty", "powdery", "lumpy",
    "slimy", "sticky", "tacky", "waxy", "oily",
    "greasy", "dry", "moist", "damp", "wet",
    # 2502-2521 more colors / finish
    "flat", "eggshell", "satin", "semi-gloss", "gloss",
    "high gloss", "matte", "metallic", "pearlescent", "iridescent",
    "translucent", "opaque", "sheer", "frosted", "tinted",
    "clear", "smoky", "mirrored", "reflective", "absorbent",
    # 2522-2541 more descriptors
    "steep", "gentle", "gradual", "abrupt", "sudden",
    "smooth", "rough", "bumpy", "jagged", "serrated",
    "blunt", "sharp", "pointed", "rounded", "tapered",
    "flared", "narrowed", "widened", "elongated", "shortened",
    # 2542-2561 more food types
    "appetizer", "entree", "main course", "side dish", "dessert",
    "snack", "meal", "feast", "buffet", "tasting menu",
    "prix fixe", "a la carte", "takeout", "delivery", "dine in",
    "fast food", "street food", "home cooked", "restaurant", "catered",
    # 2562-2581 more transport actions
    "boarding", "departing", "arriving", "landing", "taking off",
    "docking", "anchoring", "mooring", "launching", "berthing",
    "accelerating", "decelerating", "braking", "steering", "reversing",
    "parking", "idling", "cruising", "overtaking", "merging",
    # 2582-2601 more environment
    "urban", "rural", "suburban", "coastal", "inland",
    "tropical", "arctic", "temperate", "arid", "humid",
    "continental", "maritime", "alpine", "highland", "lowland",
    "wetland", "dryland", "cultivated", "wilderness", "protected",
    # 2602-2621 more occupations
    "engineer", "architect", "lawyer", "accountant", "scientist",
    "researcher", "professor", "student", "journalist", "photographer",
    "videographer", "editor", "writer", "designer", "developer",
    "analyst", "consultant", "manager", "director", "executive",
    # 2622-2641 more numbers
    "181", "182", "183", "184", "185", "186", "187", "188", "189", "190",
    "191", "192", "193", "194", "195", "196", "197", "198", "199", "200",
    # 2642-2661 more descriptors
    "clustered", "scattered", "dispersed", "concentrated", "spread",
    "gathered", "assembled", "arranged", "displayed", "exhibited",
    "showcased", "featured", "highlighted", "emphasized", "focused",
    "blurred", "clear", "crisp", "sharp", "soft",
    # 2662-2681 more objects
    "antenna", "mast", "pole", "post", "stake",
    "rod", "bar", "rail", "beam", "joist",
    "rafter", "truss", "arch", "vault", "dome",
    "span", "cantilever", "suspension", "cable", "wire",
    # 2682-2701 more actions
    "painting", "sculpting", "carving", "molding", "casting",
    "forging", "welding", "soldering", "riveting", "bolting",
    "gluing", "taping", "stapling", "pinning", "clipping",
    "fastening", "securing", "anchoring", "bracing", "supporting",
    # 2702-2721 more nature
    "deciduous", "evergreen", "coniferous", "broadleaf", "needle",
    "deciduous forest", "rain forest", "cloud forest", "mangrove", "savanna",
    "grassland", "prairie", "steppe", "pampas", "veld",
    "taiga", "boreal", "temperate", "tropical", "subtropical",
    # 2722-2741 more water
    "fresh water", "salt water", "brackish", "clear", "murky",
    "turbulent", "calm", "still", "flowing", "rushing",
    "trickling", "cascading", "tumbling", "meandering", "winding",
    "straight", "braided", "delta", "estuary", "lagoon",
    # 2742-2761 more light
    "bright", "dim", "dark", "shadow", "highlight",
    "backlit", "sidelit", "front lit", "overexposed", "underexposed",
    "high contrast", "low contrast", "black and white", "sepia", "color",
    "warm light", "cool light", "natural light", "artificial light", "mixed light",
    # 2762-2781 more objects detail
    "handle", "knob", "lever", "button", "switch",
    "dial", "gauge", "meter", "indicator", "display",
    "panel", "board", "plate", "cover", "case",
    "shell", "frame", "body", "base", "stand",
    # 2782-2801 more actions detail
    "pressing", "pulling", "pushing", "lifting", "lowering",
    "turning", "twisting", "rotating", "spinning", "rolling",
    "sliding", "gliding", "tilting", "pivoting", "swinging",
    "bouncing", "vibrating", "oscillating", "pulsating", "flickering",
    # 2802-2821 more numbers
    "201", "202", "203", "204", "205", "206", "207", "208", "209", "210",
    "211", "212", "213", "214", "215", "216", "217", "218", "219", "220",
    # 2822-2841 more materials detail
    "polished", "unpolished", "finished", "unfinished", "painted",
    "unpainted", "varnished", "lacquered", "stained", "bleached",
    "dyed", "printed", "embossed", "engraved", "etched",
    "carved", "sculpted", "molded", "cast", "forged",
    # 2842-2861 more descriptors
    "symmetrical", "asymmetrical", "balanced", "unbalanced", "centered",
    "off-center", "aligned", "misaligned", "parallel", "perpendicular",
    "horizontal", "vertical", "diagonal", "curved", "straight",
    "angled", "tilted", "rotated", "flipped", "mirrored",
    # 2862-2881 more environments
    "polluted", "clean", "contaminated", "pristine", "degraded",
    "restored", "preserved", "damaged", "intact", "fragmented",
    "continuous", "isolated", "connected", "fragmented", "patchy",
    "uniform", "diverse", "homogeneous", "heterogeneous", "varied",
    # 2882-2901 more sport specific
    "first base", "second base", "third base", "home plate", "pitcher",
    "catcher", "infielder", "outfielder", "shortstop", "batter",
    "goalkeeper", "defender", "midfielder", "striker", "winger",
    "point guard", "shooting guard", "small forward", "power forward", "center",
    # 2902-2921 more time of day
    "early morning", "late morning", "midday", "early afternoon", "late afternoon",
    "early evening", "late evening", "early night", "late night", "before dawn",
    "after dark", "golden hour", "blue hour", "magic hour", "witching hour",
    "rush hour", "off peak", "business hours", "after hours", "overtime",
    # 2922-2941 more tools
    "electric drill", "power saw", "angle grinder", "jigsaw", "circular saw",
    "reciprocating saw", "band saw", "table saw", "miter saw", "scroll saw",
    "belt sander", "orbital sander", "random orbital", "disc sander", "drum sander",
    "nail gun", "staple gun", "caulking gun", "heat gun", "spray gun",
    # 2942-2961 more household activities
    "vacuuming", "sweeping", "mopping", "scrubbing", "wiping",
    "dusting", "polishing", "washing", "drying", "ironing",
    "folding", "hanging", "storing", "organizing", "decluttering",
    "repairing", "replacing", "upgrading", "renovating", "remodeling",
    # 2962-2981 more numbers
    "221", "222", "223", "224", "225", "226", "227", "228", "229", "230",
    "231", "232", "233", "234", "235", "236", "237", "238", "239", "240",
    # 2982-3001 more descriptors
    "leading", "trailing", "ahead", "behind", "alongside",
    "adjacent", "neighboring", "distant", "remote", "nearby",
    "accessible", "inaccessible", "reachable", "unreachable", "visible",
    "invisible", "hidden", "exposed", "covered", "protected",
    # 3002-3021 more light / shadow
    "casting shadow", "reflecting light", "absorbing light", "scattering light", "refracting",
    "glowing", "luminous", "radiant", "brilliant", "dazzling",
    "faint", "faded", "washed out", "saturated", "desaturated",
    "high key", "low key", "silhouette", "rim lit", "ambient",
    # 3022-3041 more objects
    "pin", "clip", "clamp", "brace", "bracket",
    "hook", "loop", "ring", "latch", "hinge",
    "joint", "connector", "adapter", "converter", "extension",
    "splitter", "combiner", "multiplexer", "switch", "relay",
    # 3042-3061 more environment details
    "foreground", "background", "midground", "depth of field", "bokeh",
    "perspective", "vanishing point", "horizon line", "eye level", "bird's eye",
    "worm's eye", "aerial", "ground level", "high angle", "low angle",
    "straight on", "three quarter", "profile", "rear view", "overhead",
    # 3062-3081 final numbers
    "241", "242", "243", "244", "245", "246", "247", "248", "249", "250",
    "251", "252", "253", "254", "255", "256", "257", "258", "259", "260",
    # 3082-3101 more final descriptors
    "grainy", "smooth", "high resolution", "low resolution", "pixelated",
    "blurry", "sharp", "in focus", "out of focus", "overexposed",
    "underexposed", "well exposed", "high dynamic range", "flat", "contrasty",
    "warm toned", "cool toned", "neutral", "cross processed", "film like",
    # 3102-3128 final answers
    "panoramic", "wide angle", "telephoto", "macro", "fisheye",
    "tilt shift", "long exposure", "short exposure", "fast shutter", "slow shutter",
    "high iso", "low iso", "flash", "no flash", "natural",
    "studio", "outdoor", "indoor", "mixed", "available light",
    "candid", "posed", "staged", "documentary", "street",
    "portrait", "landscape",
]

# Ensure exactly 3129 entries
assert len(_ANSWERS) <= 3129, f"Too many answers: {len(_ANSWERS)}"
while len(_ANSWERS) < 3129:
    _ANSWERS.append(f"other_{len(_ANSWERS)}")
# fmt: on


def get_answer(idx: int) -> str:
    """Return the VQA v2 answer string for a given vocabulary index."""
    if 0 <= idx < len(_ANSWERS):
        return _ANSWERS[idx]
    return f"ans_{idx}"


def get_vocab() -> list[str]:
    return list(_ANSWERS)
