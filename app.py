import os, json, re, tempfile
import numpy as np
from PIL import Image
import gradio as gr
from gtts import gTTS
from ultralytics import YOLO
import torch
import asyncio
import sys
import socket


# ── Load model ────────────────────────────────────────────────────────────────
WEIGHTS_PATH = os.environ.get('WEIGHTS_PATH', 'best.pt')
LABELS_PATH  = os.environ.get('LABELS_PATH',  'class_labels.json')

model        = None
class_names  = []
idx_to_class = {}

def load_model():
    global model, class_names, idx_to_class
    if os.path.exists(WEIGHTS_PATH) and os.path.exists(LABELS_PATH):
        try:
            model = YOLO(WEIGHTS_PATH)
            model.overrides['verbose'] = False
            with open(LABELS_PATH) as f:
                raw = json.load(f)
            idx_to_class = {int(k): v for k, v in raw.items()}
            class_names  = [idx_to_class[i] for i in sorted(idx_to_class)]
        except Exception as e:
            print(f"❌ Error loading model: {e}")

load_model()

# ── Enhanced Species Info ─────────────────────────────────────────────────────
SPECIES_INFO = {
    'antelope':     {
        'habitat':'African Savanna', 'type':'Mammal', 'status':'Least Concern', 'emoji':'🦌',
        'fact':'Can run up to 90 km/h, making it one of the fastest land animals.',
        'lifespan':'10–25 years', 'diet':'Herbivore (grass, leaves, shoots)',
        'avg_weight':'20–900 kg depending on species', 'avg_length':'1.0–2.4 m body length',
        'age_clues':'Juveniles have softer, more uniform coats; males develop horns after 6–12 months; horn ring count can indicate age in some species.',
        'young_name':'Calf', 'social':'Herd animal'},
    'badger':       {
        'habitat':'Forests & Grasslands', 'type':'Mammal', 'status':'Least Concern', 'emoji':'🦡',
        'fact':'Badgers are excellent diggers and can move large amounts of soil quickly.',
        'lifespan':'5–10 years (wild), up to 14 years (captivity)', 'diet':'Omnivore (earthworms, insects, fruits, small mammals)',
        'avg_weight':'7–13 kg', 'avg_length':'60–90 cm',
        'age_clues':'Cubs are born hairless; adult badgers develop the distinctive black-and-white face markings by 2–3 months; older individuals show more worn teeth.',
        'young_name':'Cub', 'social':'Family groups (setts)'},
    'bat':          {
        'habitat':'Caves & Forests', 'type':'Mammal', 'status':'Least Concern', 'emoji':'🦇',
        'fact':'The only mammal capable of sustained flight; uses echolocation to navigate.',
        'lifespan':'10–30 years (species-dependent)', 'diet':'Insectivore / frugivore / nectarivore (varies by species)',
        'avg_weight':'2–1,000 g', 'avg_length':'3–40 cm body length',
        'age_clues':'Pups are born hairless; adult wing membranes become more leathery with age; epiphyseal fusion of finger bones marks transition to adulthood (~3–6 months).',
        'young_name':'Pup', 'social':'Colony (roost)'},
    'bear':         {
        'habitat':'Forests & Mountains', 'type':'Mammal', 'status':'Vulnerable', 'emoji':'🐻',
        'fact':'Bears have an exceptional sense of smell, 7 times better than a bloodhound.',
        'lifespan':'20–30 years (wild), up to 50 years (captivity)', 'diet':'Omnivore (fish, berries, insects, mammals)',
        'avg_weight':'100–700 kg (species-dependent)', 'avg_length':'1.2–2.8 m',
        'age_clues':'Cubs have a rounder face and shorter snout; adults develop a pronounced shoulder hump; fur grizzles with age; tooth wear and cementum annuli are precise age indicators.',
        'young_name':'Cub', 'social':'Solitary (except mothers with cubs)'},
    'bee':          {
        'habitat':'Meadows & Gardens', 'type':'Insect', 'status':'Near Threatened', 'emoji':'🐝',
        'fact':'A single bee visits up to 5,000 flowers in one day to collect nectar.',
        'lifespan':'Worker: 6 weeks; Queen: up to 5 years', 'diet':'Herbivore (pollen, nectar)',
        'avg_weight':'0.1–0.3 g', 'avg_length':'1.5–3 cm',
        'age_clues':'Young workers have brighter, less worn wing edges; older forager bees show ragged wing tips and faded body hair from repeated flights.',
        'young_name':'Larva / Pupa', 'social':'Colony (hive)'},
    'beetle':       {
        'habitat':'Global', 'type':'Insect', 'status':'Least Concern', 'emoji':'🪲',
        'fact':'Beetles make up 25% of all animal species on Earth.',
        'lifespan':'Weeks to several years (species-dependent)', 'diet':'Varies widely — herbivore, predator, or detritivore',
        'avg_weight':'Milligrams to ~100 g (Goliath beetle)', 'avg_length':'0.3 mm to 17 cm',
        'age_clues':'Larvae progress through instar stages; adults do not grow; wing cover (elytra) wear indicates age after emergence.',
        'young_name':'Larva (grub)', 'social':'Mostly solitary'},
    'bison':        {
        'habitat':'North American Plains', 'type':'Mammal', 'status':'Near Threatened', 'emoji':'🦬',
        'fact':'Despite their size, bison can run at speeds up to 56 km/h.',
        'lifespan':'15–20 years', 'diet':'Herbivore (grasses, sedges)',
        'avg_weight':'400–900 kg', 'avg_length':'2.1–3.5 m',
        'age_clues':'Calves are reddish-orange at birth, turning dark brown by ~3 months; horn curvature and mass increase with age; mature bulls develop a massive shoulder hump by age 5–6.',
        'young_name':'Calf', 'social':'Herd'},
    'boar':         {
        'habitat':'Forests & Shrublands', 'type':'Mammal', 'status':'Least Concern', 'emoji':'🐗',
        'fact':'Wild boars are highly adaptable and can thrive in many different environments.',
        'lifespan':'8–10 years (wild)', 'diet':'Omnivore (roots, fruits, small animals)',
        'avg_weight':'50–200 kg', 'avg_length':'90–200 cm',
        'age_clues':'Piglets have striped coats that fade after ~3 months; tusk size increases with age; adult males develop a tough skin shield (cutis) on the shoulders.',
        'young_name':'Piglet (or shoat)', 'social':'Sounder (female-led group)'},
    'butterfly':    {
        'habitat':'Gardens & Meadows', 'type':'Insect', 'status':'Varies', 'emoji':'🦋',
        'fact':'Butterflies taste with their feet, using chemoreceptors on their tarsi.',
        'lifespan':'Adult: days to ~1 year (migratory species like Monarch)', 'diet':'Herbivore/nectarivore (nectar, plant sap)',
        'avg_weight':'0.1–3 g', 'avg_length':'Wingspan 1–30 cm',
        'age_clues':'Freshly emerged adults have vivid, intact wing scales; older individuals show scale loss, fading colors, and wing nicks or tears.',
        'young_name':'Caterpillar (larva)', 'social':'Mostly solitary'},
    'cat':          {
        'habitat':'Global (Domestic)', 'type':'Mammal', 'status':'Domesticated', 'emoji':'🐈',
        'fact':'Cats spend 70% of their lives sleeping and have 32 muscles in each ear.',
        'lifespan':'12–18 years', 'diet':'Carnivore (meat, fish)',
        'avg_weight':'3–5 kg', 'avg_length':'46–76 cm body (not tail)',
        'age_clues':'Kittens have blue eyes that change color by 12 weeks; adult teeth fully in by 7 months; muscle tone and coat luster decline after 10 years; cloudy lens is common in seniors (12+).',
        'young_name':'Kitten', 'social':'Semi-social (tolerates other cats)'},
    'caterpillar':  {
        'habitat':'Forests & Gardens', 'type':'Insect', 'status':'Least Concern', 'emoji':'🐛',
        'fact':'Caterpillars increase their body mass by 1,000 times before metamorphosis.',
        'lifespan':'Weeks to months (larval stage only)', 'diet':'Herbivore (leaves, plant matter)',
        'avg_weight':'0.1–10 g', 'avg_length':'1–15 cm',
        'age_clues':'Caterpillars grow through a series of instar stages (typically 4–5), molting their skin each time; larger body size and more developed color patterns indicate later instars.',
        'young_name':'Larva (caterpillar is itself the larval stage)', 'social':'Mostly solitary (some species are gregarious)'},
    'chimpanzee':   {
        'habitat':'Tropical Africa', 'type':'Mammal', 'status':'Endangered', 'emoji':'🐒',
        'fact':'Chimpanzees share 98.7% of their DNA with humans.',
        'lifespan':'40–50 years (wild), up to 70 (captivity)', 'diet':'Omnivore (fruit, insects, meat)',
        'avg_weight':'32–60 kg', 'avg_length':'63–94 cm upright',
        'age_clues':'Infants have a white tail tuft (lost ~5 years); juveniles are smaller and rely on mothers; adults develop larger brow ridges and calloused skin on hands/feet; older chimps develop grey fur.',
        'young_name':'Infant', 'social':'Community (fission-fusion)'},
    'cockroach':    {
        'habitat':'Global', 'type':'Insect', 'status':'Least Concern', 'emoji':'🪳',
        'fact':'Cockroaches can survive up to a week without their head.',
        'lifespan':'1–2 years', 'diet':'Omnivore (decaying organic matter, food scraps)',
        'avg_weight':'0.1–30 g (species-dependent)', 'avg_length':'1.5–8 cm',
        'age_clues':'Nymphs are smaller and wingless; they develop wings after the final molt; adult cockroaches are fully winged (in most species); darker coloration and full adult size indicate maturity.',
        'young_name':'Nymph', 'social':'Gregarious (aggregate in harborages)'},
    'cow':          {
        'habitat':'Farmlands', 'type':'Mammal', 'status':'Domesticated', 'emoji':'🐄',
        'fact':'Cows have a nearly 360-degree panoramic vision and can see almost all around them.',
        'lifespan':'18–22 years', 'diet':'Herbivore (grass, hay, silage)',
        'avg_weight':'400–900 kg', 'avg_length':'2.0–2.5 m',
        'age_clues':'Calves lack horns and have a softer muzzle; ring count on horns (annuli) provides a rough age guide; teeth wear patterns are used by vets — calves have milk teeth, permanent teeth emerge by age 4–5.',
        'young_name':'Calf', 'social':'Herd'},
    'coyote':       {
        'habitat':'North America', 'type':'Mammal', 'status':'Least Concern', 'emoji':'🐺',
        'fact':'Coyotes are highly adaptable and have expanded their range into cities.',
        'lifespan':'6–8 years (wild)', 'diet':'Omnivore (rodents, rabbits, fruit, carrion)',
        'avg_weight':'7–21 kg', 'avg_length':'75–90 cm body',
        'age_clues':'Pups have rounder faces and softer fur; eyes open ~2 weeks; adult coat turns grizzled yellow-grey; tooth wear and eruption pattern used by biologists for age estimates.',
        'young_name':'Pup', 'social':'Family pack'},
    'crab':         {
        'habitat':'Coastal Waters', 'type':'Crustacean', 'status':'Least Concern', 'emoji':'🦀',
        'fact':'Crabs communicate by drumming or waving their claws.',
        'lifespan':'1–30 years (species-dependent)', 'diet':'Omnivore (algae, mollusks, worms, detritus)',
        'avg_weight':'0.05–3.6 kg (species-dependent)', 'avg_length':'1–45 cm carapace width',
        'age_clues':'Young crabs are tiny and go through multiple zoea larval stages before settling; carapace size increases with each molt; older crabs have thicker, more barnacled shells and worn claws.',
        'young_name':'Zoea (larva) / Megalopa', 'social':'Mostly solitary (some species aggregate)'},
    'crow':         {
        'habitat':'Global', 'type':'Bird', 'status':'Least Concern', 'emoji':'🐦',
        'fact':'Crows are among the most intelligent birds and can use tools.',
        'lifespan':'7–8 years (wild), up to 20 (captivity)', 'diet':'Omnivore (insects, carrion, seeds, small animals)',
        'avg_weight':'300–600 g', 'avg_length':'45–53 cm body',
        'age_clues':'Fledglings have a pinkish gape at the corners of the bill; juveniles have duller, brownish-black plumage; adults are glossy black; eye color shifts from blue-grey (juvenile) to dark brown/black (adult) within the first year.',
        'young_name':'Chick / Fledgling', 'social':'Family groups / large winter roosts'},
    'deer':         {
        'habitat':'Forests & Meadows', 'type':'Mammal', 'status':'Least Concern', 'emoji':'🦌',
        'fact':'Male deer grow and shed their antlers every year.',
        'lifespan':'10–20 years', 'diet':'Herbivore (grass, leaves, bark, berries)',
        'avg_weight':'15–300 kg (species-dependent)', 'avg_length':'0.9–2.0 m',
        'age_clues':'Fawns have spotted coats (fades by ~3–4 months); bucks develop their first antlers (spikes) at 1.5 years; antler mass and tine count increase until prime age (~5–7 years); older bucks show sagging belly and greying muzzle.',
        'young_name':'Fawn', 'social':'Loose herds (females) / solitary (bucks off-season)'},
    'dog':          {
        'habitat':'Global (Domestic)', 'type':'Mammal', 'status':'Domesticated', 'emoji':'🐕',
        'fact':'Dogs have a sense of smell 10,000 to 100,000 times more sensitive than humans.',
        'lifespan':'10–13 years (varies hugely by breed)', 'diet':'Omnivore (commercial food, meat, vegetables)',
        'avg_weight':'1–90 kg (breed-dependent)', 'avg_length':'15 cm – 1 m (breed-dependent)',
        'age_clues':'Puppies have a softer, rounder muzzle; teeth erupt at 3–6 weeks, permanent by 7 months; grey muzzle typically appears after age 7–8; cataracts and reduced muscle tone in seniors (10+).',
        'young_name':'Puppy', 'social':'Pack / family'},
    'dolphin':      {
        'habitat':'Oceans & Seas', 'type':'Mammal', 'status':'Least Concern', 'emoji':'🐬',
        'fact':'Dolphins sleep with one eye open and half their brain active.',
        'lifespan':'20–50 years (species-dependent)', 'diet':'Carnivore (fish, squid)',
        'avg_weight':'40–6,000 kg (bottlenose ~200 kg)', 'avg_length':'1.2–9.5 m',
        'age_clues':'Calves are darker at birth and closely follow their mother; growth layer groups (GLGs) in teeth, visible under microscope, give precise age; older dolphins develop more scars on dorsal fin.',
        'young_name':'Calf', 'social':'Pod'},
    'donkey':       {
        'habitat':'Farmlands & Deserts', 'type':'Mammal', 'status':'Domesticated', 'emoji':'🫏',
        'fact':'Donkeys have a near 360-degree field of vision due to large eyes.',
        'lifespan':'25–30 years', 'diet':'Herbivore (grass, hay, shrubs)',
        'avg_weight':'80–480 kg', 'avg_length':'0.9–1.4 m at shoulder',
        'age_clues':'Foals are long-legged with a shorter, fluffy coat; permanent teeth replace milk teeth by age 4–5; tooth wear and Galvayne\'s groove on incisors are used by vets; older donkeys show sunken above eyes and greying muzzle.',
        'young_name':'Foal', 'social':'Herd / bonded pairs'},
    'dragonfly':    {
        'habitat':'Near Water', 'type':'Insect', 'status':'Least Concern', 'emoji':'🪷',
        'fact':'Dragonflies catch their prey mid-air with a 95% success rate.',
        'lifespan':'Adult: weeks to months; nymph stage: 1–5 years', 'diet':'Carnivore (mosquitoes, gnats, small insects)',
        'avg_weight':'0.03–1 g', 'avg_length':'2–13 cm body',
        'age_clues':'Nymphs are aquatic and grow through multiple instars; newly emerged adults have soft, pale wings that harden and darken over hours; older adults show wing wear and faded color.',
        'young_name':'Nymph (naiad)', 'social':'Mostly solitary (territorial)'},
    'duck':         {
        'habitat':'Wetlands & Ponds', 'type':'Bird', 'status':'Least Concern', 'emoji':'🦆',
        'fact':'Ducks have waterproof feathers thanks to oil from their preen gland.',
        'lifespan':'5–10 years (wild)', 'diet':'Omnivore (aquatic plants, insects, small fish)',
        'avg_weight':'0.7–1.5 kg (mallard)', 'avg_length':'50–65 cm body',
        'age_clues':'Ducklings are covered in yellow-brown down; juvenile plumage is dull; adult males (drakes) develop bright breeding plumage in their first autumn; bill color and nail wear indicate age in older birds.',
        'young_name':'Duckling', 'social':'Flock / pair during breeding'},
    'eagle':        {
        'habitat':'Mountains & Coasts', 'type':'Bird', 'status':'Least Concern', 'emoji':'🦅',
        'fact':'Eagles can spot prey from up to 3 km away with their exceptional eyesight.',
        'lifespan':'20–30 years', 'diet':'Carnivore (fish, small mammals, carrion)',
        'avg_weight':'0.5–7.5 kg', 'avg_length':'66 cm – 1 m body',
        'age_clues':'Eaglets are covered in white down; juvenile plumage is mottled brown; bald eagles don\'t develop their iconic white head until 4–5 years; eye color lightens from brown to yellow with age.',
        'young_name':'Eaglet', 'social':'Mostly solitary (pairs during breeding)'},
    'elephant':     {
        'habitat':'Africa & Asia', 'type':'Mammal', 'status':'Endangered', 'emoji':'🐘',
        'fact':'Elephants are the only animals that cannot jump.',
        'lifespan':'60–70 years', 'diet':'Herbivore (grasses, leaves, bark, fruit)',
        'avg_weight':'2,700–6,000 kg', 'avg_length':'5–7 m body',
        'age_clues':'Calves are small, fuzzy, and stay close to their mother; tusk growth is a rough guide — older elephants have longer, more worn tusks; ear vein patterns and skin wrinkling increase with age; molar progression (6 sets through lifetime) provides age bracket.',
        'young_name':'Calf', 'social':'Matriarchal herd'},
    'flamingo':     {
        'habitat':'Tropical Wetlands', 'type':'Bird', 'status':'Least Concern', 'emoji':'🦩',
        'fact':'Flamingos are pink because of carotenoid pigments in their food.',
        'lifespan':'20–30 years (wild)', 'diet':'Omnivore (algae, brine shrimp, crustaceans)',
        'avg_weight':'2–4 kg', 'avg_length':'1.1–1.45 m tall',
        'age_clues':'Chicks hatch grey-white and do not develop pink plumage until 2–3 years; juvenile plumage is grey-brown; adults are vivid pink; deeper pink coloration indicates better diet and health, not necessarily older age.',
        'young_name':'Chick', 'social':'Large colony'},
    'fly':          {
        'habitat':'Global', 'type':'Insect', 'status':'Least Concern', 'emoji':'🪰',
        'fact':'Flies have compound eyes giving them a nearly 360-degree view.',
        'lifespan':'Adult housefly: 28 days on average', 'diet':'Omnivore/detritivore (decaying matter, sugars, excrement)',
        'avg_weight':'~12 mg (housefly)', 'avg_length':'0.5–6 cm (species-dependent)',
        'age_clues':'Larvae (maggots) go through 3 instars before pupation; adult age is difficult to assess visually but wing wear and faded bristle color indicate older individuals.',
        'young_name':'Maggot (larva)', 'social':'Mostly solitary (swarm near food sources)'},
    'fox':          {
        'habitat':'Forests & Suburban areas', 'type':'Mammal', 'status':'Least Concern', 'emoji':'🦊',
        'fact':'Foxes use the Earth\'s magnetic field to judge distance when pouncing on prey.',
        'lifespan':'2–5 years (wild), up to 14 years (captivity)', 'diet':'Omnivore (rodents, rabbits, fruit, insects)',
        'avg_weight':'2–14 kg', 'avg_length':'45–90 cm body',
        'age_clues':'Kits have rounded muzzles and dark, fuzzy fur; adult coat turns the characteristic reddish-orange; older foxes show faded or greying fur and more prominent facial scarring.',
        'young_name':'Kit / Cub', 'social':'Family group'},
    'giraffe':      {
        'habitat':'African Savanna', 'type':'Mammal', 'status':'Vulnerable', 'emoji':'🦒',
        'fact':'Giraffes sleep less than 2 hours a day and have the highest blood pressure of any mammal.',
        'lifespan':'20–27 years', 'diet':'Herbivore (acacia leaves, fruit, grass)',
        'avg_weight':'700–1,930 kg', 'avg_length':'4.3–5.7 m tall',
        'age_clues':'Calves are ~1.8 m tall at birth; coat patterns darken with age; horn-like ossicones are small and hair-covered in calves, flatter and more pronounced in adults; older bulls develop calcified crown ossicones.',
        'young_name':'Calf', 'social':'Loose herd (tower)'},
    'goat':         {
        'habitat':'Mountains & Farms', 'type':'Mammal', 'status':'Domesticated', 'emoji':'🐐',
        'fact':'Goats have rectangular pupils that give them a wide field of vision.',
        'lifespan':'15–18 years', 'diet':'Herbivore (grass, shrubs, leaves)',
        'avg_weight':'20–140 kg', 'avg_length':'0.9–1.4 m body',
        'age_clues':'Kids are smaller with softer, shorter horns; annual rings on horns indicate age; teeth eruption pattern (kids have milk teeth; permanent incisors emerge between 1–4 years) is a reliable age indicator.',
        'young_name':'Kid', 'social':'Herd'},
    'goldfish':     {
        'habitat':'Freshwater', 'type':'Fish', 'status':'Least Concern', 'emoji':'🐟',
        'fact':'Goldfish have a memory span of several months, not seconds.',
        'lifespan':'10–15 years (up to 25 years in captivity)', 'diet':'Omnivore (algae, insects, small crustaceans)',
        'avg_weight':'0.1–3 kg (wild carp form can be larger)', 'avg_length':'5–45 cm',
        'age_clues':'Scale rings (annuli) can be counted like tree rings under a microscope; juveniles are typically darker/olive colored, brightening with age; fin ray cross-sections also reveal age bands.',
        'young_name':'Fry', 'social':'Shoal'},
    'goose':        {
        'habitat':'Wetlands & Parks', 'type':'Bird', 'status':'Least Concern', 'emoji':'🪿',
        'fact':'Geese mate for life and will stay with an injured partner.',
        'lifespan':'10–24 years (species-dependent)', 'diet':'Herbivore (grass, aquatic plants, grain)',
        'avg_weight':'2–8 kg', 'avg_length':'55–110 cm body',
        'age_clues':'Goslings are covered in yellow-green down; juvenile plumage is duller than adults; adult plumage fully develops by first winter; bill and leg color intensify with age in some species.',
        'young_name':'Gosling', 'social':'Flock / mated pair'},
    'gorilla':      {
        'habitat':'Central Africa', 'type':'Mammal', 'status':'Critically Endangered', 'emoji':'🦍',
        'fact':'Gorillas share 98.3% of their DNA with humans.',
        'lifespan':'35–40 years (wild)', 'diet':'Herbivore/omnivore (leaves, fruit, insects)',
        'avg_weight':'68–200 kg', 'avg_length':'1.25–1.8 m upright',
        'age_clues':'Infants are small and cling to their mother\'s back; juveniles are playful and smaller; males develop the distinctive "silverback" (grey saddle hair) after age 12.',
        'young_name':'Infant', 'social':'Troop (silverback-led)'},
    'grasshopper':  {
        'habitat':'Meadows & Fields', 'type':'Insect', 'status':'Least Concern', 'emoji':'🦗',
        'fact':'Grasshoppers can jump 20 times their own body length.',
        'lifespan':'Adult: 1–2 months; full cycle ~1 year', 'diet':'Herbivore (grasses, leaves, crops)',
        'avg_weight':'0.3–2.5 g', 'avg_length':'1–12 cm',
        'age_clues':'Nymphs lack fully developed wings; wing pads grow progressively through 5–6 instars; adult grasshoppers have full wings; wing wear and faded color indicate older adults.',
        'young_name':'Nymph (hopper)', 'social':'Solitary to gregarious (locust phase)'},
    'hamster':      {
        'habitat':'Deserts & Steppes', 'type':'Mammal', 'status':'Vulnerable', 'emoji':'🐹',
        'fact':'Hamsters can carry up to half their body weight in food in their cheek pouches.',
        'lifespan':'2–3 years (wild & captivity)', 'diet':'Omnivore (seeds, grains, insects)',
        'avg_weight':'30–200 g (species-dependent)', 'avg_length':'5–34 cm',
        'age_clues':'Young hamsters have a smooth coat and bright eyes; adults develop a more textured fur; older hamsters (18+ months) show reduced activity, cloudier eyes, and thinning fur.',
        'young_name':'Pup', 'social':'Solitary (highly territorial)'},
    'hare':         {
        'habitat':'Grasslands & Tundra', 'type':'Mammal', 'status':'Least Concern', 'emoji':'🐇',
        'fact':'Hares are born fully furred with open eyes, unlike rabbits.',
        'lifespan':'4–5 years (wild)', 'diet':'Herbivore (grasses, bark, buds)',
        'avg_weight':'3–7 kg', 'avg_length':'40–70 cm',
        'age_clues':'Leverets are born fully furred and mobile; young hares are smaller with a narrower eye lens weight; adults are larger; eye lens weight is the standard field technique for estimating age in harvested hares.',
        'young_name':'Leveret', 'social':'Mostly solitary'},
    'hedgehog':     {
        'habitat':'Europe & Asia', 'type':'Mammal', 'status':'Least Concern', 'emoji':'🦔',
        'fact':'Hedgehogs are immune to some snake venoms.',
        'lifespan':'3–5 years (wild), up to 10 (captivity)', 'diet':'Omnivore (insects, worms, slugs, berries)',
        'avg_weight':'400–900 g', 'avg_length':'15–30 cm',
        'age_clues':'Hoglets are born with spines under the skin (emerge within hours); juveniles are smaller with softer underbelly fur; adults have fully hardened spines; older individuals show facial greying and reduced spine density.',
        'young_name':'Hoglet', 'social':'Solitary'},
    'hippopotamus': {
        'habitat':'Sub-Saharan Africa', 'type':'Mammal', 'status':'Vulnerable', 'emoji':'🦛',
        'fact':'Despite their bulk, hippos can run faster than humans on land.',
        'lifespan':'40–50 years', 'diet':'Herbivore (grasses, aquatic plants)',
        'avg_weight':'1,500–3,200 kg', 'avg_length':'3.3–5.2 m body',
        'age_clues':'Calves are small and rosy-colored; canine tusks grow continuously and are a primary age indicator; older bulls have massive, battle-scarred heads with prominent tusk length.',
        'young_name':'Calf', 'social':'Pod (school)'},
    'hornbill':     {
        'habitat':'Tropical Forests', 'type':'Bird', 'status':'Varies', 'emoji':'🐦',
        'fact':'Hornbills are known for their distinctive large bills and casques.',
        'lifespan':'20–40 years (species-dependent)', 'diet':'Omnivore (fruit, insects, small vertebrates)',
        'avg_weight':'0.1–4 kg (species-dependent)', 'avg_length':'30 cm – 1.2 m',
        'age_clues':'Juveniles have smaller casques and duller bill coloration; casque grows progressively over the first few years; fully grown casque and vivid bill colors indicate adult status; older birds show heavier bill wear.',
        'young_name':'Chick', 'social':'Monogamous pairs / small flocks'},
    'horse':        {
        'habitat':'Grasslands', 'type':'Mammal', 'status':'Domesticated', 'emoji':'🐴',
        'fact':'Horses can sleep standing up thanks to a special stay apparatus in their legs.',
        'lifespan':'25–30 years', 'diet':'Herbivore (grass, hay, grains)',
        'avg_weight':'380–1,000 kg', 'avg_length':'1.4–1.8 m at shoulder (height)',
        'age_clues':'Foals are long-legged with a shorter, fuzzier coat; milk teeth replaced by permanent teeth by age 5; Galvayne\'s groove (groove on upper corner incisor) appears at ~10 years and is used by vets to estimate age up to ~30.',
        'young_name':'Foal', 'social':'Herd'},
    'hyena':        {
        'habitat':'African Savanna', 'type':'Mammal', 'status':'Least Concern', 'emoji':'🐾',
        'fact':'Spotted hyenas are matriarchal, with females dominating the clan.',
        'lifespan':'12–25 years (wild)', 'diet':'Carnivore/scavenger (wildebeest, zebra, carrion)',
        'avg_weight':'40–86 kg', 'avg_length':'1.2–1.8 m body',
        'age_clues':'Cubs are born with dark, almost black fur that fades to adult spotting by ~3 months; juveniles are slimmer and lighter; adults have a fully developed mane; older individuals show worn, yellowed teeth.',
        'young_name':'Cub', 'social':'Clan (matriarchal)'},
    'jellyfish':    {
        'habitat':'All Oceans', 'type':'Invertebrate', 'status':'Least Concern', 'emoji':'🪼',
        'fact':'Jellyfish have survived for 500 million years without a brain or heart.',
        'lifespan':'Hours to several months (adult medusa stage; some species are potentially immortal)', 'diet':'Carnivore (zooplankton, small fish, shrimp)',
        'avg_weight':'0.004–200+ kg (lion\'s mane)', 'avg_length':'1 cm – 2 m bell diameter (tentacles can reach 30+ m)',
        'age_clues':'Jellyfish begin as tiny polyps attached to substrate; strobilation produces ephyrae (juvenile medusae); bell size and tentacle complexity increase with age; Turritopsis dohrnii can revert to polyp stage and restart.',
        'young_name':'Ephyra (juvenile medusa)', 'social':'Bloom / solitary'},
    'kangaroo':     {
        'habitat':'Australia', 'type':'Mammal', 'status':'Least Concern', 'emoji':'🦘',
        'fact':'Kangaroos cannot walk backwards and are the only large animals that hop.',
        'lifespan':'6–8 years (wild)', 'diet':'Herbivore (grass, shrubs)',
        'avg_weight':'18–90 kg', 'avg_length':'0.8–1.8 m body',
        'age_clues':'Joeys are born the size of a jelly bean and develop in the pouch for ~8 months; sub-adults leave the pouch but stay close to the mother; males grow much larger than females; older males develop heavily muscled forearms.',
        'young_name':'Joey', 'social':'Mob'},
    'koala':        {
        'habitat':'Australia', 'type':'Mammal', 'status':'Vulnerable', 'emoji':'🐨',
        'fact':'Koalas sleep up to 22 hours a day to conserve energy from their eucalyptus diet.',
        'lifespan':'13–18 years', 'diet':'Herbivore (eucalyptus leaves)',
        'avg_weight':'4–14 kg', 'avg_length':'60–85 cm',
        'age_clues':'Joeys ride on the mother\'s back and are smaller with thinner fur; adults have a large, leathery nose pad; older males have a prominent scent gland (chest stain) and broader head.',
        'young_name':'Joey', 'social':'Solitary'},
    'ladybugs':     {
        'habitat':'Gardens & Fields', 'type':'Insect', 'status':'Least Concern', 'emoji':'🐞',
        'fact':'A ladybug\'s bright colors warn predators that they taste unpleasant.',
        'lifespan':'Adult: 1–2 years', 'diet':'Carnivore (aphids, scale insects, mites)',
        'avg_weight':'~1–30 mg', 'avg_length':'0.8–1.8 cm',
        'age_clues':'Larvae are elongated and spiny, very unlike adults; pupae are plump and immobile; newly emerged adults have pale, soft elytra that harden and deepen in color within hours; the number of spots is fixed for each individual and species — it does not change or increase with age.',
        'young_name':'Larva', 'social':'Solitary (aggregate to overwinter)'},
    'leopard':      {
        'habitat':'Africa & Asia', 'type':'Mammal', 'status':'Vulnerable', 'emoji':'🐆',
        'fact':'Leopards are the strongest climbers among big cats and carry prey into trees.',
        'lifespan':'12–17 years (wild)', 'diet':'Carnivore (ungulates, rodents, birds)',
        'avg_weight':'17–90 kg', 'avg_length':'90–190 cm body',
        'age_clues':'Cubs have fuzzy, pale coats with less defined rosettes; rosette patterns sharpen with age; adult males are notably larger than females; older individuals show scarring on face and ears.',
        'young_name':'Cub', 'social':'Solitary'},
    'lion':         {
        'habitat':'African Savanna', 'type':'Mammal', 'status':'Vulnerable', 'emoji':'🦁',
        'fact':'A lion\'s roar can be heard from 8 km away.',
        'lifespan':'10–14 years (wild)', 'diet':'Carnivore (zebra, wildebeest, buffalo)',
        'avg_weight':'120–250 kg', 'avg_length':'1.4–2.5 m body',
        'age_clues':'Cubs are spotted at birth (fading by ~3 months); mane begins growing at ~1 year and darkens with age — a dark, full mane indicates maturity (5+ years); nose pigmentation darkens progressively with age.',
        'young_name':'Cub', 'social':'Pride'},
    'lizard':       {
        'habitat':'Global', 'type':'Reptile', 'status':'Varies', 'emoji':'🦎',
        'fact':'Some lizards can detach their tail to escape predators — it grows back.',
        'lifespan':'3–20 years (species-dependent)', 'diet':'Omnivore (insects, vegetation, small animals)',
        'avg_weight':'0.5 g – 150 kg (Komodo dragon)', 'avg_length':'3 cm – 3 m (Komodo dragon)',
        'age_clues':'Hatchlings are miniature adults; lizards never stop growing; body length and girth increase with age; bone cross-sections (skeletochronology) reveal growth rings; shed frequency and scale wear hint at health and age.',
        'young_name':'Hatchling', 'social':'Mostly solitary (varies by species)'},
    'lobster':      {
        'habitat':'Ocean Floors', 'type':'Crustacean', 'status':'Least Concern', 'emoji':'🦞',
        'fact':'Lobsters can live for over 100 years and are biologically immortal.',
        'lifespan':'50–100+ years', 'diet':'Carnivore (fish, mollusks, sea urchins)',
        'avg_weight':'0.5–20 kg', 'avg_length':'25–64 cm',
        'age_clues':'Juveniles are tiny and planktonic; carapace size and weight increase with age — a 1 kg lobster is roughly 5–7 years old; eyestalk neural lipofuscin concentration is used for non-lethal age estimation.',
        'young_name':'Larva (phyllosoma)', 'social':'Mostly solitary (aggregate in crevices)'},
    'monkey':       {
        'habitat':'Tropical & Subtropical Forests', 'type':'Mammal', 'status':'Varies', 'emoji':'🐒',
        'fact':'Some monkeys, like capuchins, use tools such as stones to crack open nuts.',
        'lifespan':'15–40 years (species-dependent)', 'diet':'Omnivore (fruit, leaves, insects)',
        'avg_weight':'0.1–40 kg', 'avg_length':'14 cm – 1 m body',
        'age_clues':'Infants have brighter facial skin and cling to their mothers; juveniles are smaller and playful; adults develop more pronounced facial features and coloration specific to their species.',
        'young_name':'Infant', 'social':'Troop'},
    'mosquito':     {
        'habitat':'Global', 'type':'Insect', 'status':'Least Concern', 'emoji':'🦟',
        'fact':'Only female mosquitoes bite; they need blood to develop their eggs.',
        'lifespan':'Adult female: 2–3 weeks; male: ~1 week', 'diet':'Female: blood (for egg development) + nectar; Male: nectar only',
        'avg_weight':'~2.5 mg', 'avg_length':'3–9 mm',
        'age_clues':'Larvae are aquatic and go through 4 instars; pupae are comma-shaped and non-feeding; adult age is estimated by wing wear and ovarian dissection (parity analysis) in field research.',
        'young_name':'Larva (wriggler)', 'social':'Swarm (mating)'},
    'moth':         {
        'habitat':'Global', 'type':'Insect', 'status':'Varies', 'emoji':'🦋',
        'fact':'Moths navigate using the moon and stars, which is why they circle artificial lights.',
        'lifespan':'Adult: days to 1 month (many species don\'t eat as adults)', 'diet':'Herbivore (caterpillar stage); nectar or nothing (adult stage)',
        'avg_weight':'0.003–30 g', 'avg_length':'Wingspan 3 mm – 30 cm',
        'age_clues':'Caterpillars progress through 4–6 instars; freshly emerged adults have fully intact scales; scale loss, wing fading, and frayed edges appear rapidly in older adults.',
        'young_name':'Caterpillar (larva)', 'social':'Mostly solitary'},
    'mouse':        {
        'habitat':'Global', 'type':'Mammal', 'status':'Least Concern', 'emoji':'🐭',
        'fact':'Mice can squeeze through gaps as small as 6mm wide.',
        'lifespan':'1–3 years (wild)', 'diet':'Omnivore (seeds, grains, insects, plant matter)',
        'avg_weight':'12–30 g', 'avg_length':'6–10 cm body',
        'age_clues':'Pups are born hairless and blind; eyes open at ~2 weeks; juvenile coat is softer and grayer; adults have fuller, glossier fur; older mice show thinning fur, reduced activity, and duller eyes.',
        'young_name':'Pup (or pinkie when newborn)', 'social':'Colony'},
    'octopus':      {
        'habitat':'Oceans', 'type':'Cephalopod', 'status':'Least Concern', 'emoji':'🐙',
        'fact':'Octopuses have three hearts and blue blood.',
        'lifespan':'1–5 years', 'diet':'Carnivore (crabs, shrimp, fish)',
        'avg_weight':'0.015–10 kg (common octopus)', 'avg_length':'30 cm – 4.3 m arm span',
        'age_clues':'Hatchlings are tiny larvae (planktonic); growth rings on statoliths (balance organs) indicate age like tree rings; older individuals are larger and show more chromatophore complexity.',
        'young_name':'Hatchling', 'social':'Solitary'},
    'okapi':        {
        'habitat':'Congo Rainforest', 'type':'Mammal', 'status':'Endangered', 'emoji':'🦒',
        'fact':'Okapis are the only living relative of the giraffe.',
        'lifespan':'20–30 years', 'diet':'Herbivore (leaves, fruits, seeds)',
        'avg_weight':'200–350 kg', 'avg_length':'1.9–2.5 m body',
        'age_clues':'Calves are born with striped hindquarters similar to adults; young okapis are smaller with a more uniform coat tone; males develop small horn-like ossicones after 1 year; older individuals show more pronounced facial wrinkling.',
        'young_name':'Calf', 'social':'Solitary'},
    'orangutan':    {
        'habitat':'Borneo & Sumatra', 'type':'Mammal', 'status':'Critically Endangered', 'emoji':'🦧',
        'fact':'Orangutans share 97% of their DNA with humans.',
        'lifespan':'35–45 years', 'diet':'Omnivore (fruit, leaves, insects, bark)',
        'avg_weight':'30–90 kg', 'avg_length':'1.1–1.4 m upright',
        'age_clues':'Infants are very light-colored and cling to their mother; sub-adults are smaller; adult males develop large cheek pads (flanges) after ~15 years; older males have larger throat sacs.',
        'young_name':'Infant', 'social':'Semi-solitary'},
    'otter':        {
        'habitat':'Rivers & Coasts', 'type':'Mammal', 'status':'Varies', 'emoji':'🦦',
        'fact':'Sea otters hold hands while sleeping so they don\'t drift apart.',
        'lifespan':'10–15 years (wild)', 'diet':'Carnivore (fish, crabs, clams, sea urchins)',
        'avg_weight':'5–45 kg (sea otter up to 45 kg)', 'avg_length':'0.6–1.2 m body',
        'age_clues':'Pups have a fluffy, buoyant natal coat that is replaced at ~2–3 months; juveniles have slightly duller fur; adults have a dense, waterproof double coat; older otters show facial greying and tooth wear.',
        'young_name':'Pup', 'social':'Raft (sea otters) / family groups (river otters)'},
    'owl':          {
        'habitat':'Global Forests', 'type':'Bird', 'status':'Least Concern', 'emoji':'🦉',
        'fact':'Owls can rotate their heads up to 270 degrees.',
        'lifespan':'5–25 years (species-dependent)', 'diet':'Carnivore (rodents, insects, small birds)',
        'avg_weight':'0.04–4 kg', 'avg_length':'14–75 cm body',
        'age_clues':'Owlets are covered in white down; juvenile plumage is fluffier; adults have well-defined feather patterns; eye color can change with age (e.g., barn owls start with dark eyes).',
        'young_name':'Owlet', 'social':'Mostly solitary'},
    'ox':           {
        'habitat':'Farmlands', 'type':'Mammal', 'status':'Domesticated', 'emoji':'🐂',
        'fact':'Oxen are domesticated cattle used as draft animals for millennia.',
        'lifespan':'15–20 years', 'diet':'Herbivore (grass, hay, grain)',
        'avg_weight':'500–1,000 kg', 'avg_length':'2.0–2.8 m body',
        'age_clues':'As domesticated cattle, oxen age similarly to cows; calves have smooth skin and small or absent horns; horn ring count provides rough age estimation; permanent incisors erupt fully by age 4–5.',
        'young_name':'Calf', 'social':'Herd / working pairs'},
    'oyster':       {
        'habitat':'Coastal Waters', 'type':'Mollusk', 'status':'Least Concern', 'emoji':'🦪',
        'fact':'A single oyster can filter up to 190 liters of water per day.',
        'lifespan':'20+ years', 'diet':'Filter feeder (phytoplankton, organic particles)',
        'avg_weight':'Up to 500 g (shell included)', 'avg_length':'5–25 cm shell',
        'age_clues':'Shell growth rings (visible when cut cross-section) reflect annual growth cycles; summer rings are wider; shell ridges and thickness increase with age; older oysters have irregular, deeply cupped shells.',
        'young_name':'Spat (juvenile oyster)', 'social':'Reef / bed (aggregate)'},
    'panda':        {
        'habitat':'China Bamboo Forests', 'type':'Mammal', 'status':'Vulnerable', 'emoji':'🐼',
        'fact':'Giant pandas eat up to 12-38 kg of bamboo per day.',
        'lifespan':'15–20 years (wild)', 'diet':'Herbivore (bamboo — 99% of diet)',
        'avg_weight':'70–125 kg', 'avg_length':'1.2–1.5 m body',
        'age_clues':'Cubs are born tiny (~100 g), pink, and nearly hairless; black-and-white pattern develops over first month; adults are much larger with fully developed "eye patches"; older pandas may show faded markings.',
        'young_name':'Cub', 'social':'Solitary'},
    'parrot':       {
        'habitat':'Tropical Forests', 'type':'Bird', 'status':'Varies', 'emoji':'🦜',
        'fact':'Parrots are among the most intelligent birds and can mimic human speech.',
        'lifespan':'20–80+ years (species-dependent; macaws can live 80+ years)', 'diet':'Herbivore (seeds, nuts, fruit, buds)',
        'avg_weight':'10 g – 1.6 kg (species-dependent)', 'avg_length':'10 cm – 1 m',
        'age_clues':'Chicks are hatched featherless; juvenile plumage is duller; adult coloration fully develops between 1–4 years (varies by species); iris color deepens with age (e.g., cockatoos); DNA testing is most reliable for exact age.',
        'young_name':'Chick', 'social':'Flock / bonded pairs'},
    'pelecaniformes':{
        'habitat':'Coastal Areas', 'type':'Bird', 'status':'Least Concern', 'emoji':'🐦',
        'fact':'This order of birds includes pelicans, herons, and cormorants.',
        'lifespan':'10–25 years (varies by species)', 'diet':'Carnivore (fish, amphibians, crustaceans)',
        'avg_weight':'0.5–15 kg', 'avg_length':'45 cm – 1.8 m',
        'age_clues':'Chicks hatch with sparse down and are altricial; juvenile plumage is mottled or brown; adult breeding plumage (including pouches and crests in pelicans) develops over 2–4 years; bill pouch color and eye color may intensify in older breeding adults.',
        'young_name':'Chick', 'social':'Colony / pair'},
    'penguin':      {
        'habitat':'Antarctica & Southern Hemisphere', 'type':'Bird', 'status':'Varies', 'emoji':'🐧',
        'fact':'Penguins are flightless birds that are exceptional swimmers reaching 25+ km/h.',
        'lifespan':'6–20 years (species-dependent)', 'diet':'Carnivore (fish, squid, krill)',
        'avg_weight':'1–40 kg (species-dependent)', 'avg_length':'30 cm – 1.1 m',
        'age_clues':'Chicks are covered in grey or brown fluffy down; juvenile plumage transitions to adult coloring at 1–2 years; adult plumage is crisp black-and-white; age estimated by flipper band records in wild populations.',
        'young_name':'Chick', 'social':'Colony'},
    'pig':          {
        'habitat':'Farmlands', 'type':'Mammal', 'status':'Domesticated', 'emoji':'🐷',
        'fact':'Pigs are highly intelligent and can learn their names within weeks.',
        'lifespan':'15–20 years', 'diet':'Omnivore (grains, vegetables, roots, insects)',
        'avg_weight':'50–350 kg (breed-dependent)', 'avg_length':'0.9–1.8 m',
        'age_clues':'Piglets have striped or spotted coats at birth; permanent teeth eruption chart (known as "dentition") provides age from months to years; body size and bone density increase steadily; older pigs develop larger, drooping ears and a more pronounced snout.',
        'young_name':'Piglet', 'social':'Sounder (group)'},
    'pigeon':       {
        'habitat':'Global Urban', 'type':'Bird', 'status':'Least Concern', 'emoji':'🕊️',
        'fact':'Pigeons can recognize themselves in a mirror, a rare cognitive feat.',
        'lifespan':'3–5 years (wild), up to 15 (captivity)', 'diet':'Herbivore (seeds, grains, fruit)',
        'avg_weight':'250–600 g', 'avg_length':'29–37 cm body',
        'age_clues':'Squabs are altricial (hatched helpless); juvenile plumage is duller with a softer cere (nostril area); adults have a bright, waxy cere and iridescent neck feathers; eye color shifts from dark grey to orange-red in adults.',
        'young_name':'Squab', 'social':'Flock'},
    'porcupine':    {
        'habitat':'Africa & Americas', 'type':'Mammal', 'status':'Least Concern', 'emoji':'🦔',
        'fact':'Porcupine quills have microscopic barbs that make them difficult to remove.',
        'lifespan':'5–15 years (species-dependent)', 'diet':'Herbivore (bark, roots, tubers, fruit)',
        'avg_weight':'0.7–27 kg (species-dependent)', 'avg_length':'40–90 cm body',
        'age_clues':'Young porcupines are born with soft quills that harden within hours; juveniles are smaller; adults have dense, fully hardened quills; older individuals may show thinning quills and reduced activity.',
        'young_name':'Porcupette', 'social':'Mostly solitary / family groups'},
    'possum':       {
        'habitat':'Americas & Australia', 'type':'Mammal', 'status':'Least Concern', 'emoji':'🐾',
        'fact':'Opossums are immune to most venomous snake bites.',
        'lifespan':'1–2 years (Virginia opossum in wild)', 'diet':'Omnivore (fruits, insects, carrion, small animals)',
        'avg_weight':'0.3–6 kg', 'avg_length':'32–50 cm body',
        'age_clues':'Joeys are born tiny and develop in the pouch for ~2 months; sub-adults are smaller with proportionally larger ears; adults develop a more pointed snout and grizzled coat; tooth wear is a field indicator.',
        'young_name':'Joey', 'social':'Solitary'},
    'rabbit':       {
        'habitat':'Grasslands & Meadows', 'type':'Mammal', 'status':'Least Concern', 'emoji':'🐇',
        'fact':'Rabbits cannot vomit and their digestive system requires them to eat specialized cecotrope pellets.',
        'lifespan':'1–9 years (wild), up to 12 years (domestic)', 'diet':'Herbivore (grass, hay, vegetables)',
        'avg_weight':'0.4–2 kg (wild)', 'avg_length':'30–50 cm',
        'age_clues':'Kittens are born blind, deaf, and hairless; eyes open at ~10 days; young rabbits have softer, sleeker fur; adults have more pronounced musculature; senior rabbits (5+ years) show reduced activity and occasionally greying muzzle.',
        'young_name':'Kitten / Kit', 'social':'Colony (warren)'},
    'raccoon':      {
        'habitat':'North America', 'type':'Mammal', 'status':'Least Concern', 'emoji':'🦝',
        'fact':'Raccoons have dexterous front paws they use almost like hands.',
        'lifespan':'2–3 years (wild)', 'diet':'Omnivore (berries, insects, fish, small animals)',
        'avg_weight':'3.5–9 kg', 'avg_length':'40–70 cm body',
        'age_clues':'Kits have faint mask and ringtail patterns that develop clearly by 3 months; adults have full, dense winter coat; tooth wear and eye lens weight used by researchers for age estimation.',
        'young_name':'Kit', 'social':'Family groups'},
    'rat':          {
        'habitat':'Global', 'type':'Mammal', 'status':'Least Concern', 'emoji':'🐀',
        'fact':'Rats laugh when tickled, producing high-frequency ultrasonic sounds.',
        'lifespan':'2–3 years (wild)', 'diet':'Omnivore (grains, meat, garbage)',
        'avg_weight':'150–500 g', 'avg_length':'16–24 cm body',
        'age_clues':'Pups are born hairless and helpless; coat develops by 2 weeks; juveniles are smaller and smoother; adults are fully sized; older rats show greying fur, reduced grooming, and cataracts.',
        'young_name':'Pup (or kitten)', 'social':'Colony'},
    'reindeer':     {
        'habitat':'Arctic & Subarctic', 'type':'Mammal', 'status':'Vulnerable', 'emoji':'🦌',
        'fact':'Reindeer eyes change color from gold in summer to blue in winter.',
        'lifespan':'15–20 years', 'diet':'Herbivore (lichen, grass, sedges, willow)',
        'avg_weight':'60–300 kg', 'avg_length':'1.6–2.1 m body',
        'age_clues':'Calves are born spotted and grow rapidly; both males and females grow antlers (unique among deer); antler mass and tine complexity increase until prime age (~5–7 years); tooth wear pattern is a standard field age indicator.',
        'young_name':'Calf', 'social':'Herd'},
    'rhinoceros':   {
        'habitat':'Africa & Asia', 'type':'Mammal', 'status':'Critically Endangered', 'emoji':'🦏',
        'fact':'A rhino\'s horn is made of keratin, the same protein as human nails.',
        'lifespan':'35–50 years', 'diet':'Herbivore (grass, leaves, fruit)',
        'avg_weight':'800–2,300 kg', 'avg_length':'3–4 m',
        'age_clues':'Calves lack horns at birth (they develop within months); horn length increases steadily; skin fold patterns deepen with age; older rhinos show heavier skin and more wrinkles.',
        'young_name':'Calf', 'social':'Mostly solitary'},
    'sandpiper':    {
        'habitat':'Shorelines', 'type':'Bird', 'status':'Least Concern', 'emoji':'🐦',
        'fact':'Sandpipers migrate thousands of kilometers between breeding and wintering grounds.',
        'lifespan':'8–15 years', 'diet':'Carnivore (invertebrates, insects, small crustaceans)',
        'avg_weight':'15–200 g (species-dependent)', 'avg_length':'12–60 cm',
        'age_clues':'Juveniles have fresh, neat-edged feathers with pale buff or rufous fringing; adults in non-breeding plumage are plainer gray-brown; breeding plumage (streaked/spotted rufous) develops in spring; bill wear and feather abrasion increase with age.',
        'young_name':'Chick', 'social':'Flock (migratory) / pairs (breeding)'},
    'seahorse':     {
        'habitat':'Shallow Oceans', 'type':'Fish', 'status':'Vulnerable', 'emoji':'🐟',
        'fact':'Male seahorses carry and give birth to young.',
        'lifespan':'1–5 years', 'diet':'Carnivore (tiny crustaceans, plankton)',
        'avg_weight':'0.003–0.5 kg', 'avg_length':'1.5–35 cm',
        'age_clues':'Juveniles are miniature adults; size is the primary indicator — body length increases steadily; coronet (bony crown) shape matures with age; fin ray cross-sections provide precise age via growth rings.',
        'young_name':'Fry (called "babies" colloquially)', 'social':'Monogamous pairs'},
    'seal':         {
        'habitat':'Polar Regions', 'type':'Mammal', 'status':'Least Concern', 'emoji':'🦭',
        'fact':'Seals can sleep underwater, surfacing automatically to breathe.',
        'lifespan':'20–30 years (species-dependent)', 'diet':'Carnivore (fish, squid, crustaceans)',
        'avg_weight':'55–2,400 kg (species-dependent)', 'avg_length':'1.2–5 m',
        'age_clues':'Pups have a fluffy, pale natal coat (lanugo) shed after weaning; juveniles are slimmer and lighter; adults develop pronounced musculature; tooth cementum layers (annuli) are the gold standard for age determination.',
        'young_name':'Pup', 'social':'Colony (haul-out) / solitary at sea'},
    'shark':        {
        'habitat':'All Oceans', 'type':'Fish', 'status':'Varies', 'emoji':'🦈',
        'fact':'Sharks have been around for 450 million years, predating the dinosaurs.',
        'lifespan':'20–500 years (species-dependent; Greenland shark can live 400+ years)', 'diet':'Carnivore (fish, marine mammals, squid)',
        'avg_weight':'7 kg – 18,000 kg (whale shark)', 'avg_length':'0.7–12 m',
        'age_clues':'Young sharks (pups) are proportionally slimmer; vertebral growth rings can be counted like tree rings for age; larger body size generally indicates greater age in well-studied species.',
        'young_name':'Pup', 'social':'Mostly solitary (some schooling species)'},
    'sheep':        {
        'habitat':'Farmlands', 'type':'Mammal', 'status':'Domesticated', 'emoji':'🐑',
        'fact':'Sheep have excellent memories and can recognize up to 50 other sheep faces.',
        'lifespan':'10–12 years', 'diet':'Herbivore (grass, hay, clover)',
        'avg_weight':'45–160 kg', 'avg_length':'1.2–1.8 m body',
        'age_clues':'Lambs are small with soft, fluffy wool; milk teeth (8 incisors) present up to age 1; permanent incisors erupt in pairs from age 1–4, allowing farmers to identify "1-tooth", "2-tooth" up to "full mouth" (4-year-old) sheep; wool texture coarsens with age.',
        'young_name':'Lamb', 'social':'Flock'},
    'snake':        {
        'habitat':'Global', 'type':'Reptile', 'status':'Varies', 'emoji':'🐍',
        'fact':'Snakes smell with their tongues, using it to sense air molecules.',
        'lifespan':'10–30 years (species-dependent)', 'diet':'Carnivore (rodents, birds, eggs, insects)',
        'avg_weight':'10 g – 100 kg (reticulated python)', 'avg_length':'10 cm – 7.5 m',
        'age_clues':'Hatchlings are fully formed but smaller; snakes never stop growing; body girth and length increase with age; scale wear and shedding frequency can hint at overall health/age; eye clarity changes.',
        'young_name':'Hatchling / Snakelet', 'social':'Mostly solitary'},
    'sparrow':      {
        'habitat':'Global', 'type':'Bird', 'status':'Least Concern', 'emoji':'🐦',
        'fact':'House sparrows are one of the most widespread birds on Earth.',
        'lifespan':'3–5 years (wild)', 'diet':'Omnivore (seeds, insects, scraps)',
        'avg_weight':'24–40 g', 'avg_length':'14–18 cm body',
        'age_clues':'Juveniles have duller, streaked plumage without the black bib of adult males; first-year birds have a yellow gape at the bill corners; adult males develop a striking black bib and chestnut head markings; feather wear and eye ring color distinguish age classes.',
        'young_name':'Chick / Fledgling', 'social':'Flock'},
    'squid':        {
        'habitat':'All Oceans', 'type':'Cephalopod', 'status':'Least Concern', 'emoji':'🦑',
        'fact':'Squids can change color in milliseconds using chromatophores.',
        'lifespan':'1–5 years (most species are short-lived)', 'diet':'Carnivore (fish, crustaceans, other squid)',
        'avg_weight':'0.03–500 kg (giant squid)', 'avg_length':'15 cm – 13 m (giant squid)',
        'age_clues':'Hatchlings are miniature adults; statoliths (balance organs) contain daily growth rings that can be counted; mantle length increases throughout life; larger body size generally = greater age.',
        'young_name':'Paralarva (hatchling)', 'social':'School / solitary'},
    'squirrel':     {
        'habitat':'Forests & Parks', 'type':'Mammal', 'status':'Least Concern', 'emoji':'🐿️',
        'fact':'Squirrels plant thousands of trees by forgetting where they buried their nuts.',
        'lifespan':'6–12 years (wild)', 'diet':'Herbivore (nuts, seeds, berries, fungi)',
        'avg_weight':'100–900 g (species-dependent)', 'avg_length':'15–30 cm body',
        'age_clues':'Kittens are born hairless and blind; juveniles have softer, fluffier fur; adults have a full, bushy tail; tooth eruption and wear patterns are used in field studies for age estimation.',
        'young_name':'Kitten (or kit)', 'social':'Mostly solitary (tree squirrels) / colonial (ground squirrels)'},
    'starfish':     {
        'habitat':'Oceans', 'type':'Echinoderm', 'status':'Least Concern', 'emoji':'⭐',
        'fact':'Starfish can regenerate lost arms; some species can grow a full body from one arm.',
        'lifespan':'5–35 years (species-dependent)', 'diet':'Carnivore (clams, mussels, oysters)',
        'avg_weight':'0.03–5 kg', 'avg_length':'5–65 cm arm span',
        'age_clues':'Juveniles are smaller and may have proportionally shorter arms; growth rings visible in skeletal ossicles (microscopic) reflect annual cycles; arm width and overall size increase with age.',
        'young_name':'Bipinnaria larva (juvenile sea star)', 'social':'Mostly solitary'},
    'swan':         {
        'habitat':'Lakes & Rivers', 'type':'Bird', 'status':'Least Concern', 'emoji':'🦢',
        'fact':'Swans mate for life and can live for over 20 years.',
        'lifespan':'20–30 years', 'diet':'Herbivore (aquatic plants, algae, grass)',
        'avg_weight':'7–15 kg', 'avg_length':'1.0–1.7 m body',
        'age_clues':'Cygnets are grey-brown and fluffy; juvenile plumage is brownish-white for the first year; adults develop pure white plumage after the first full molt (~1–2 years); bill coloration (orange with black base) fully develops in adults; older birds have a more prominent knob at the base of the bill.',
        'young_name':'Cygnet', 'social':'Pair / family group'},
    'tiger':        {
        'habitat':'Asia', 'type':'Mammal', 'status':'Endangered', 'emoji':'🐯',
        'fact':'No two tigers have the same stripe pattern — it\'s unique like a fingerprint.',
        'lifespan':'10–15 years (wild)', 'diet':'Carnivore (deer, boar, buffalo)',
        'avg_weight':'75–300 kg', 'avg_length':'1.4–2.9 m body',
        'age_clues':'Cubs have a softer, less saturated coat; adult stripes are bold and well-defined; older tigers develop a more angular face, wear on canine teeth, and loose facial skin; males are notably larger than females.',
        'young_name':'Cub', 'social':'Solitary (except mothers with cubs)'},
    'turkey':       {
        'habitat':'North America', 'type':'Bird', 'status':'Least Concern', 'emoji':'🦃',
        'fact':'Wild turkeys can fly at speeds of up to 88 km/h over short distances.',
        'lifespan':'3–5 years (wild)', 'diet':'Omnivore (seeds, insects, berries, small reptiles)',
        'avg_weight':'3.6–11 kg', 'avg_length':'70–125 cm body',
        'age_clues':'Poults are small and downy; juvenile males (jakes) have a short, rounded tail with longer central feathers; adult toms develop a full fan-shaped tail and long beard of feathers; spur length on the legs increases with age — short spurs indicate juveniles, long curved spurs indicate older males.',
        'young_name':'Poult', 'social':'Flock'},
    'turtle':       {
        'habitat':'Global Waters & Land', 'type':'Reptile', 'status':'Varies', 'emoji':'🐢',
        'fact':'Sea turtles navigate using the Earth\'s magnetic field to return to their birthplace.',
        'lifespan':'40–100+ years (sea turtles)', 'diet':'Herbivore or omnivore (algae, jellyfish, crabs)',
        'avg_weight':'0.2 kg – 500 kg (leatherback)', 'avg_length':'10 cm – 2.1 m',
        'age_clues':'Hatchlings are much smaller than adults; scute growth rings on the shell (skeletochronology) can estimate age; shell length is a reliable proxy for age class; older turtles often have barnacle growth on the shell.',
        'young_name':'Hatchling', 'social':'Mostly solitary'},
    'whale':        {
        'habitat':'All Oceans', 'type':'Mammal', 'status':'Varies', 'emoji':'🐋',
        'fact':'Blue whales are the largest animals to have ever existed on Earth.',
        'lifespan':'20–200 years (bowhead whale may live 200+ years)', 'diet':'Carnivore (krill, fish, squid)',
        'avg_weight':'1,000–150,000 kg', 'avg_length':'4–30 m',
        'age_clues':'Calves are darker and closely follow their mother; wax plug layers in the ear canal (earplug laminae) count like tree rings; barnacle and scar accumulation on skin increases with age.',
        'young_name':'Calf', 'social':'Pod / solitary'},
    'wolf':         {
        'habitat':'Forests & Tundra', 'type':'Mammal', 'status':'Least Concern', 'emoji':'🐺',
        'fact':'Wolves can travel up to 200 km in a single day when hunting.',
        'lifespan':'6–8 years (wild)', 'diet':'Carnivore (deer, elk, moose, small mammals)',
        'avg_weight':'23–80 kg', 'avg_length':'1.0–1.6 m body',
        'age_clues':'Pups have rounder heads, shorter muzzles, and blue eyes (changing to yellow/amber by 3–4 months); adults have sharper features; tooth wear and eruption are key indicators used in field research; older wolves have grizzled fur.',
        'young_name':'Pup', 'social':'Pack'},
    'wombat':       {
        'habitat':'Australia', 'type':'Mammal', 'status':'Least Concern', 'emoji':'🐾',
        'fact':'Wombats produce cube-shaped droppings — the only known animal to do so.',
        'lifespan':'15–20 years', 'diet':'Herbivore (grasses, roots, bark)',
        'avg_weight':'17–35 kg', 'avg_length':'85–130 cm body',
        'age_clues':'Joeys develop in the pouch for ~6–7 months; sub-adults are smaller; adults are compact and powerfully built; older wombats develop more worn incisor teeth and a broader, flatter head.',
        'young_name':'Joey', 'social':'Mostly solitary (share burrow systems)'},
    'woodpecker':   {
        'habitat':'Forests', 'type':'Bird', 'status':'Least Concern', 'emoji':'🐦',
        'fact':'Woodpeckers peck up to 20 times per second without getting a headache.',
        'lifespan':'5–11 years (wild)', 'diet':'Omnivore (insects in wood, berries, tree sap)',
        'avg_weight':'7 g – 600 g (species-dependent)', 'avg_length':'15–58 cm',
        'age_clues':'Juveniles have duller plumage and less prominent markings; the red cap or other distinctive coloration of adults is absent or reduced in juveniles; bill shows less wear in first-year birds; adults develop fully by first breeding season (~1 year).',
        'young_name':'Chick', 'social':'Mostly solitary / pairs during breeding'},
    'zebra':        {
        'habitat':'Africa', 'type':'Mammal', 'status':'Near Threatened', 'emoji':'🦓',
        'fact':'Every zebra\'s stripe pattern is unique, like a human fingerprint.',
        'lifespan':'20–30 years', 'diet':'Herbivore (grass, leaves)',
        'avg_weight':'175–450 kg', 'avg_length':'2.0–2.5 m body',
        'age_clues':'Foals are born with brownish stripes that whiten over time; adults have well-defined black-and-white stripes; older zebras have more prominent ridges along the face and show dental wear.',
        'young_name':'Foal', 'social':'Herd (harem)'},
     
}

# ── Improved Fallback Logic ──────────────────────────────────────────────────
def get_species_info(name: str) -> dict:
    key = name.lower().strip().replace("_", " ")
    # Check for direct match
    if key in SPECIES_INFO:
        return SPECIES_INFO[key]
    # Check for partial match (e.g., "red fox" matches "fox")
    for k in SPECIES_INFO:
        if k in key:
            return SPECIES_INFO[k]
    # Intelligent default if still not found
    return {
        'habitat': 'Worldwide', 'type': 'Animal', 'status': 'Unknown',
        'emoji': '🐾', 'fact': 'A fascinating species found in various ecosystems.',
        'lifespan': 'Varies', 'diet': 'Varies', 'avg_weight': 'N/A', 'avg_length': 'N/A',
        'age_clues': 'Visual age markers vary by subspecies.', 'young_name': 'Juvenile', 'social': 'Unknown',
    }
# ── Predict & Logic ───────────────────────────────────────────────────────────
def predict_image(pil_img, top_k=5):
    if model is None:
        return None, "Model not loaded. Check weights.", "Error", None

    result = model(pil_img, verbose=False)[0]
    if result.probs is None:
        return None, "Model is not a classification model.", "Error", None

    probs = result.probs.data.cpu().numpy()
    names = result.names
    top_idx = probs.argsort()[::-1][:top_k]
    
    # 1. Clean the predicted name
    raw_name = names[top_idx[0]]
    top_conf = float(probs[top_idx[0]]) * 100
    clean_target = raw_name.lower().strip().replace("_", " ")

    # 2. SMART SEARCH (Matches "squirrel" even in "grey_squirrel")
    info = None
    for key in SPECIES_INFO:
        if key in clean_target:
            info = SPECIES_INFO[key]
            break
            
    if not info:
        info = {
            'habitat': 'Global', 'type': 'Animal', 'status': 'Unknown',
            'emoji': '🐾', 'fact': 'Species data is currently being indexed.',
            'lifespan': 'Varies', 'diet': 'Varies', 'avg_weight': 'N/A', 
            'avg_length': 'N/A', 'age_clues': 'Visual age markers vary.', 
            'young_name': 'Juvenile', 'social': 'Unknown'
        }

    # 3. Build Detailed Markdown Info
    md_output = f"""
# {info['emoji']} {clean_target.upper()} 
**Confidence Score:** `{top_conf:.1f}%`

---

### 🧬 Biological Overview
- **Type:** {info['type']} | **Status:** {info['status']}
- **Social:** {info.get('social', 'Unknown')} | **Young:** {info.get('young_name', 'Young')}

### 🗺️ Habitat & Lifestyle
- **Home:** {info['habitat']}
- **Diet:** {info.get('diet', 'Unknown')}
- **Lifespan:** {info.get('lifespan', 'N/A')}

### 📏 Physical Profile
- **Weight:** {info.get('avg_weight', 'N/A')}
- **Size:** {info.get('avg_length', 'N/A')}

### 🕰️ Age Markers
{info.get('age_clues', 'Age estimation clues not available.')}

---
> **FUN FACT:** {info['fact']}
"""

    # 4. Build HTML Progress Bars
    html_bars = "<div style='padding: 10px;'>"
    for idx in top_idx:
        n = names[idx].title().replace("_", " ")
        conf = float(probs[idx]) * 100
        html_bars += f"""
        <div style="margin-bottom:15px;">
            <div style="display:flex; justify-content:space-between; margin-bottom:5px; font-weight:bold; color:#34d399;">
                <span>{n}</span><span>{conf:.1f}%</span>
            </div>
            <div style="background:rgba(255,255,255,0.1); height:8px; border-radius:4px;">
                <div style="background:linear-gradient(90deg, #059669, #34d399); width:{conf}%; height:100%; border-radius:4px; box-shadow:0 0 10px #34d39988;"></div>
            </div>
        </div>
        """
    html_bars += "</div>"

    speech_text = f"Identification complete. This is a {clean_target}. {info['fact']}"

    return html_bars, md_output, f"{info['emoji']} {clean_target.title()}", speech_text
def run(image):
    if image is None: return None, "Please upload an image.", "", ""
    # Since type="pil" in gr.Image, 'image' is already a PIL object
    return predict_image(image.convert("RGB"))

def generate_voice_guide(speech_text):
    if not speech_text:
        return None
    try:
        tts = gTTS(text=speech_text, lang='en')
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_file.name)
        return temp_file.name
    except Exception:
        return None

def capture_image_state(image):
    if image is None:
        return None, "<div style='font-weight:600;'>Ready to scan.</div>", gr.update(interactive=False)
    return image, "<div style='color:#34d399; font-weight:700;'>✅ Image ready. Click Identify Animal.</div>", gr.update(interactive=True)

def prepare_result_page(image):
    if image is None:
        return (
            "<div style='color:#fbbf24; font-weight:700;'>Please upload / capture / paste an image first.</div>",
            gr.update(visible=True),
            gr.update(visible=False),
            None,
            "<div style='color:#f8fafc;'>Prediction confidence will appear here.</div>",
            "Species details will appear here after identification.",
            "",
            "",
            gr.update(value=None),
        )

    return (
        "<div style='color:#34d399; font-weight:700;'>🔎 Processing image...</div>",
        gr.update(visible=False),
        gr.update(visible=True),
        image,
        "<div style='color:#f8fafc;'>Processing prediction...</div>",
        "Processing species details...",
        "Processing...",
        "",
        gr.update(value=None),
    )

def identify_controller(image):
    # UI-safe wrapper: never navigate to result page without content
    if image is None:
        return (
            "<div style='color:#fbbf24; font-weight:700;'>Please upload / capture / paste an image first.</div>",
            gr.update(visible=True),
            gr.update(visible=False),
            None,  # result_image
            "<div style='color:#f8fafc;'>Prediction confidence will appear here.</div>",
            "Species details will appear here after identification.",
            "",
            "",
        )

    html_bars, md_output, top_name, speech_text = run(image)
    return (
        "<div style='color:#34d399; font-weight:700;'>✅ Scan complete. Showing results.</div>",
        gr.update(visible=False),
        gr.update(visible=True),
        image,  # result_image
        html_bars or "<div style='color:#f8fafc;'>No predictions returned.</div>",
        md_output or "No species info returned.",
        top_name or "",
        speech_text or "",
    )

# ── UI Design ─────────────────────────────────────────────────────────────────
custom_css = """
/* Global Gradio color tokens override */
:root,
.gradio-container {
    --body-background-fill: #020617 !important;
    --background-fill-primary: rgba(15, 23, 42, 0.9) !important;
    --background-fill-secondary: rgba(15, 23, 42, 0.82) !important;
    --block-background-fill: rgba(15, 23, 42, 0.78) !important;
    --block-border-color: rgba(16, 185, 129, 0.25) !important;
    --input-background-fill: rgba(2, 6, 23, 0.96) !important;
    --input-border-color: rgba(16, 185, 129, 0.35) !important;
}

/* Deep Midnight Background */
.gradio-container {
    background: radial-gradient(circle at center, #0f172a, #020617) !important;
    color: #e2e8f0 !important;
}

.gradio-container * {
    color: #e2e8f0 !important;
}

.gradio-container p,
.gradio-container span,
.gradio-container label,
.gradio-container legend,
.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container h4,
.gradio-container h5,
.gradio-container h6 {
    color: #e2e8f0 !important;
}

.gradio-container .prose,
.gradio-container .prose *,
.gradio-container [data-testid="block-label"],
.gradio-container [data-testid="block-label"] *,
.gradio-container .wrap .label-wrap,
.gradio-container .wrap .label-wrap *,
.gradio-container .gr-label,
.gradio-container .gr-label * {
    color: #f8fafc !important;
}

.gradio-container .markdown-body,
.gradio-container .markdown-body *,
.gradio-container .rendered-markdown,
.gradio-container .rendered-markdown * {
    color: #f8fafc !important;
}

/* Glowing Emerald Cards */
.custom-panel {
    background: rgba(30, 41, 59, 0.7) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(16, 185, 129, 0.3) !important;
    border-radius: 20px !important;
    box-shadow: 0 0 20px rgba(0, 0, 0, 0.5) !important;
    padding: 25px !important;
}
body {
    background: #020617 !important; /* Deep dark blue */
}

.card {
    background: rgba(15, 23, 42, 0.8) !important;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(0, 229, 160, 0.3) !important; /* Emerald glow border */
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    border-radius: 20px;
    padding: 18px !important;
    margin-bottom: 14px !important;
}

.btn-primary {
    background: linear-gradient(135deg, #10b981, #059669) !important;
    box-shadow: 0 0 15px rgba(16, 185, 129, 0.4);
    border: none;
}
/* Neon Emerald Button */
button.primary {
    background: linear-gradient(135deg, #10b981, #059669) !important;
    border: none !important;
    box-shadow: 0 0 15px rgba(16, 185, 129, 0.4) !important;
    transition: all 0.3s ease !important;
}
button.primary:hover {
    transform: scale(1.02) !important;
    box-shadow: 0 0 25px rgba(16, 185, 129, 0.6) !important;
}

/* Force dark surfaces (no white panels) */
.gradio-container .block,
.gradio-container .gr-box,
.gradio-container .gr-panel,
.gradio-container .gr-form,
.gradio-container .panel-wrap,
.gradio-container .wrap,
.gradio-container .form,
.gradio-container .container,
.gradio-container .gradio-html,
.gradio-container .gradio-markdown,
.gradio-container .gradio-image,
.gradio-container .gradio-audio {
    background: rgba(15, 23, 42, 0.75) !important;
    border-color: rgba(16, 185, 129, 0.25) !important;
}

.gradio-container input,
.gradio-container textarea,
.gradio-container select {
    background: rgba(2, 6, 23, 0.95) !important;
    color: #e2e8f0 !important;
    border: 1px solid rgba(16, 185, 129, 0.35) !important;
}

/* Dark upload/image area + readable text */
#scan-image,
#scan-image > div,
#scan-image .wrap,
#scan-image .image-container,
#scan-image .image-frame,
#scan-image .upload-container,
#scan-image .empty {
    background: rgba(2, 6, 23, 0.96) !important;
    color: #e2e8f0 !important;
    border-color: rgba(16, 185, 129, 0.35) !important;
    max-width: 320px !important;
    margin: 0 auto !important;
}

#result-image,
#result-image > div,
#result-image .wrap,
#result-image .image-container,
#result-image .image-frame,
#result-image .upload-container,
#result-image .empty {
    background: rgba(2, 6, 23, 0.96) !important;
    color: #e2e8f0 !important;
    border-color: rgba(16, 185, 129, 0.35) !important;
    max-width: 320px !important;
    margin: 0 auto !important;
}

/* Scanning effect over uploaded image panel */
#scan-image {
    position: relative;
    overflow: hidden;
    max-width: 320px !important;
    margin: 0 auto !important;
}

#scan-image::before {
    content: "";
    position: absolute;
    left: 0;
    right: 0;
    top: -6px;
    height: 4px;
    background: linear-gradient(90deg, transparent, #34d399, transparent);
    box-shadow: 0 0 14px #34d399;
    animation: imageScanMove 2s linear infinite;
    z-index: 8;
    pointer-events: none;
}

@keyframes imageScanMove {
    0% { top: 0; opacity: 0.35; }
    50% { opacity: 1; }
    100% { top: calc(100% - 6px); opacity: 0.35; }
}

"""

with gr.Blocks() as demo:
    voice_text_state = gr.State("")
    captured_image_state = gr.State(None)
    with gr.Column(visible=True) as scan_page:
        gr.HTML("<h1 style='text-align:center; color:#10b981; text-shadow: 0 0 10px #10b981;'>🐾 ANIMAL SPECIES IDENTIFIER</h1>")
        gr.HTML(
            "<div style='text-align:center; color:#f8fafc; font-size:18px; margin-bottom:14px;'>"
            "Animal species identifier"
            "<br/>"
            "<span style='color:#cbd5e1;'>"
            "Ultralytics YOLO classification using local weights (<code>best.pt</code>) + species knowledge base + automatic voice guide via gTTS."
            "</span>"
            "<br/>"
            "<span style='color:#94a3b8; font-size:14px;'>Upload, capture, or paste an animal image to scan and identify species. Recommended runtime: Python 3.11.9</span>"
            "</div>"
        )
        with gr.Column(elem_classes="custom-panel"):
            img_input = gr.Image(
                label="Upload / Camera / Paste Image",
                type="pil",
                sources=["upload", "webcam", "clipboard"],
                elem_id="scan-image",
                height=320,
                width=320,
            )
            identify_btn = gr.Button("🔍 Identify Animal", variant="primary", interactive=False)
            scan_status = gr.HTML("<div style='font-weight:600;'>Ready to scan.</div>")

    with gr.Column(visible=False) as result_page:
        gr.HTML("<h2 style='text-align:center; color:#10b981;'>📊 Scan Result</h2>")
        with gr.Column(elem_classes="custom-panel"):
            with gr.Row():
                with gr.Column(elem_classes="card"):
                    result_image = gr.Image(label="Uploaded Image", type="pil", elem_id="result-image", height=320, width=320)
                with gr.Column(elem_classes="card"):
                    gr.HTML("<div style='color:#f8fafc; font-weight:700;'>Prediction</div>")
                    name_out = gr.Textbox(label="Top Identification", interactive=False)
                    top_preds_html = gr.HTML("<div style='color:#f8fafc;'>Prediction confidence will appear here.</div>")
            with gr.Row():
                with gr.Column(elem_classes="card"):
                    gr.HTML("<div style='color:#f8fafc; font-weight:700;'>Info</div>")
                    result_md = gr.Markdown("Species details will appear here after identification.")
                with gr.Column(elem_classes="card"):
                    gr.HTML("<div style='color:#f8fafc; font-weight:700;'>Voice</div>")
                    audio_out = gr.Audio(label="Voice Guide", type="filepath")
            back_btn = gr.Button("🔄 Scan Another Image", variant="primary")

    img_input.change(
        fn=capture_image_state,
        inputs=[img_input],
        outputs=[captured_image_state, scan_status, identify_btn],
    )

    (
        identify_btn.click(
            fn=prepare_result_page,
            inputs=[captured_image_state],
            outputs=[
                scan_status,
                scan_page,
                result_page,
                result_image,
                top_preds_html,
                result_md,
                name_out,
                voice_text_state,
                audio_out,
            ],
        )
        .then(
            fn=identify_controller,
            inputs=[captured_image_state],
            outputs=[
                scan_status,
                scan_page,
                result_page,
                result_image,
                top_preds_html,
                result_md,
                name_out,
                voice_text_state,
            ],
        )
        .then(
            fn=generate_voice_guide,
            inputs=[voice_text_state],
            outputs=[audio_out],
        )
    )

    back_btn.click(
        fn=lambda: (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(value=None),
            None,
            "<div style='font-weight:600;'>Ready to scan.</div>",
            gr.update(interactive=False),
            gr.update(value=None),  # result_image
            gr.update(value="<div style='color:#f8fafc;'>Prediction confidence will appear here.</div>"),
            gr.update(value="Species details will appear here after identification."),
            gr.update(value=""),
            "",
            gr.update(value=None),
        ),
        outputs=[scan_page, result_page, img_input, captured_image_state, scan_status, identify_btn, result_image, top_preds_html, result_md, name_out, voice_text_state, audio_out]
    )

if __name__ == "__main__":
    demo.queue()
    demo.launch(
        theme=gr.themes.Soft(primary_hue="emerald"),
        css=custom_css,
    )