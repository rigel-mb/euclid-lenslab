"""Self-contained visual explanation of the matched-neighbour experiment."""
from pathlib import Path
import base64
import html
import json

ROOT = Path(__file__).resolve().parents[2]


def gallery(root):
    """Embed the fixed examples and all ten graph neighbours, including missing images."""
    payload = json.loads((Path(root) / 'results/neighbour_gallery.json').read_text())
    options = ''.join(f'<option value="{i}">Paire {i+1} · candidate de grade {p["grade"]}</option>'
                      for i, p in enumerate(payload['pairs']))
    encoded = json.dumps(payload, ensure_ascii=False).replace('<', '\\u003c')
    return f'''<div class="gallery" id="gallery">
<div class="gallery-heading"><p class="eyebrow">VOIR LA COMPARAISON</p><h3>Une image de départ. Ses dix voisines.</h3>
<p>Choisis une paire : la candidate et sa référence servent chacune de point de départ. Les labels indiquent ensuite combien de leurs voisines sont des candidates A, B ou C.</p></div>
<div class="gallery-controls"><label for="gallery-pair">Paire affichée <select id="gallery-pair">{options}</select></label>
<label class="label-toggle"><input id="gallery-labels" type="checkbox" checked> Afficher les labels et les comptes</label></div>
<div class="gallery-legend" id="gallery-legend"><span class="grade-A">Candidate A</span><span class="grade-B">Candidate B</span><span class="grade-C">Candidate C</span><span class="grade-U">Sans label de candidate</span></div>
<div id="gallery-views" aria-live="polite"></div>
<noscript><p>Active JavaScript pour changer de paire. La planche complète reste visible ci-dessous.</p></noscript>
<p class="gallery-note"><strong>Trois paires tirées à l’avance, une par grade.</strong> Elles ne sont pas sélectionnées pour leur contraste. Les dix voisins sont ceux du calcul sur les 2 252 objets, dans l’ordre de distance Zoobot. Une image manquante reste à sa place. Le résultat global se lit sous la galerie.</p>
<p class="gallery-note">Aperçus VIS de 9,6″, nord en haut, est à gauche ; barre de 2″. Contraste ajusté par image. Ces aperçus servent à l’inspection : les distances viennent des coordonnées Zoobot publiées, calculées en amont avec VIS + Y. Clique sur une vignette pour l’agrandir.</p>
</div><script type="application/json" id="gallery-data">{encoded}</script>'''


def build(root=ROOT):
    root = Path(root)
    def read(name): return json.loads((root/'results'/name).read_text())
    s = read('expanded_sampling_summary.json')
    a = read('neighbour_summary.json')
    d, e = read('second_comparison.json'), read('second_encoder_summary.json')
    p = a['primary']; z, dn = d['zoobot'], d['dino']
    c = s['candidate_flow']
    def num(v): return f'{int(v):,}'.replace(',', '\u202f')
    def pct(v): return f'{100*v:.1f}'.replace('.', ',')+' %'
    def pts(v): return f'{100*v:.1f}'.replace('.', ',')
    def fig(name, caption, hero=False):
        payload=base64.b64encode((root/'figures'/f'{name}.png').read_bytes()).decode()
        tag=f'<img src="data:image/png;base64,{payload}" alt="{html.escape(caption,quote=True)}" loading="lazy">'
        return tag if hero else f'<figure>{tag}<figcaption>{caption}</figcaption></figure>'
    def note(title, text): return f'<aside class="take"><b>{title}</b><p>{text}</p></aside>'
    def details(title, text): return f'<details><summary>{title}</summary>{text}</details>'
    body=f'''<body>
<header><a class="brand" href="#top">LensLab <span>/ Euclid Q1</span></a><nav><a href="#data">Données</a><a href="#pairs">Paires</a><a href="#neighbours">Voisins</a><a href="#comparison">DINOv2</a></nav></header>
<main id="top">
<section class="hero"><div><p class="eyebrow">MOHAMED BOUREGA · DATA SCIENTIST &amp; AI ENGINEER</p>
<h1>Les voisines<br>d’une candidate<br><em>sont-elles aussi<br>des candidates ?</em></h1>
<p class="lead">Comparer les images proches d’une candidate lentille connue aux images proches d’un objet comparable sans label.</p>
<div class="tags"><span>Données scientifiques</span><span>Représentations d’images</span><span>Analyse de voisinage</span></div></div>
<div class="space">{fig('eda_hero','Euclid VIS : image réelle d’une candidate publiée de grade A.',True)}<small>IMAGE EUCLID · ESA / EUCLID CONSORTIUM</small></div></section>
<div class="intro"><p>Une galaxie massive peut déformer la lumière d’une galaxie plus lointaine et produire des arcs : c’est une lentille gravitationnelle. Dans Euclid, des experts ont déjà évalué des candidates.</p><p><strong>Mon objectif :</strong> vérifier si les représentations Zoobot rapprochent ces candidates entre elles, même face à des références comparables en brillance et en taille. Ici, « voisin » désigne une ressemblance pour le modèle.</p></div>
<div class="stats overview"><div><strong>{num(c['final_objects'])}</strong><span>objets représentés</span></div><div><strong>{num(c['retained_candidates'])}</strong><span>paires candidate / référence</span></div><div><strong>10</strong><span>voisins examinés par objet</span></div></div>
<a class="gallery-jump" href="#gallery">Explorer les images et leurs voisines →</a>
<p class="small">Les figures s’agrandissent au clic. Les détails du protocole se déplient à la demande.</p>
<section id="data"><div class="heading"><span>01</span><div><h2>Des candidates connues, des objets sans label.</h2><p>Comprendre la sélection avant de comparer les images.</p></div></div>
<div class="cards"><article><h3>Les candidates A, B et C</h3><p>Le catalogue publié contient 2 415 identifiants uniques. Les grades des experts traduisent différents niveaux de confiance ; toutes ces candidates ne sont pas des lentilles confirmées.</p></article><article><h3>Les références</h3><p>Ce sont des objets absents de la liste de candidates. Leur statut réel reste inconnu : « sans label de candidate » ne signifie pas « non-lentille ».</p></article></div>
{fig('eda_gallery','Deux images par grade, prises dans la sélection aléatoire fixée pour l’exploration. Contraste ajusté individuellement pour lire les formes.')}
{fig('eda_measurements','Le catalogue de candidates est comparé à 10 000 objets de contexte issus de trois régions du ciel. Ces populations ont des sélections différentes ; ce contexte ne remplace pas l’appariement.')}
<p class="small">Le jeu exact S1/S2 et les négatifs DESI utilisés pour entraîner le détecteur du papier C n’ont pas été récupérés pour ce projet. J’étudie donc le voisinage de représentations préentraînées, sans entraîner de détecteur sur nos labels.</p>
</section>
<section id="pairs"><div class="heading"><span>02</span><div><h2>Une référence comparable pour chaque candidate.</h2><p>Contrôler les différences évidentes avant de mesurer la ressemblance des images.</p></div></div>
<div class="criteria"><div><b>Brillance apparente</b><span>Écart de magnitude ≤0,3</span></div><div><b>Surface apparente</b><span>Écart de log₁₀ aire ≤0,2</span></div><div><b>Région du ciel</b><span>Distance ≤108″</span></div></div>
<p class="small">Parmi les références admissibles encore disponibles, on choisit la meilleure correspondance selon ces critères, sans réutiliser d’objet. <strong>Ce choix utilise les mesures du catalogue, pas la ressemblance calculée par Zoobot.</strong></p>
{fig('cohort_flow','Chaque exclusion est comptée. Tous les objets retenus possèdent des coordonnées dans le même fichier Zoobot publié.')}
{fig('expanded_balance','L’appariement équilibre la brillance et la surface. Le signal sur bruit, non utilisé pour choisir les paires, reste proche entre les deux groupes.')}
{details('Les critères exacts et les exclusions',f'''<p>Les deux objets doivent avoir une détection VIS, une aire de segmentation ≥300 pixels et une magnitude totale AB &lt;22,5. La référence doit aussi être à ≥20″ de toute position de candidate publiée. Les écarts autorisés restent fixes ; l’ordre des candidates est fixé par une graine. C’est une allocation gloutonne reproductible, sans garantie d’optimalité globale.</p><p>Sur 2 415 candidates, 999 n’ont pas de représentation publiée, une autre échoue aux coupes communes, puis 289 n’ont aucune référence admissible. On conserve donc <b>1 126 candidates et 1 126 références</b> : 112 candidates A, 131 B et 883 C. Le ratio de 50 % est construit pour cette comparaison ; ce n’est pas la fréquence des lentilles dans Euclid.</p><p>Des différences de couleur, de morphologie et des corrélations spatiales peuvent subsister malgré l’appariement.</p>''')}
</section>
<section id="neighbours"><div class="heading"><span>03</span><div><h2>Comparer les voisins dans l’espace Zoobot.</h2><p>Mesurer les ressemblances, puis regarder les labels.</p></div></div>
<p class="small">Zoobot a appris à décrire la morphologie des galaxies. Les auteurs ont publié <strong>40 coordonnées par objet</strong>, obtenues par réduction de ses caractéristiques d’images. J’utilise ces coordonnées telles quelles pour rechercher les dix objets les plus proches.</p>
<div class="experiment" aria-label="Schéma de la méthode">
<div class="experiment-start"><b>Les 2 252 objets dans le même espace Zoobot</b><span>Distances entre représentations, sans utiliser les grades</span></div>
<div class="experiment-branches"><div><b>Pour chaque candidate</b><span>Ses 10 voisins les plus proches</span><strong>{pct(p['candidate_neighbour_fraction'])}</strong><span>de candidates parmi les voisins</span></div><div><b>Pour sa référence</b><span>Ses 10 voisins les plus proches</span><strong>{pct(p['reference_neighbour_fraction'])}</strong><span>de candidates parmi les voisins</span></div></div>
<div class="experiment-end"><b>Comparer les deux fractions dans chaque paire</b><span>Écart moyen : +{pts(p['difference'])} points</span></div></div>
<p class="small"><strong>Dans les deux recherches, l’objet lui-même et son partenaire sont exclus.</strong> Les voisins viennent du corpus entier ; ils ne sont pas nécessairement proches dans le ciel.</p>
{gallery(root)}
{details('Voir les trois paires sur une seule planche',fig('neighbour_gallery','Les mêmes exemples que dans la galerie interactive : chaque image de départ est suivie de ses dix voisins exacts. Les références restent sans label de candidate, y compris lorsqu’elles sont proches de candidates.'))}
{fig('neighbours','Moyenne de la proportion de candidates dans les listes de dix voisins. Le niveau de référence de 50 % vient de la composition du corpus.')}
{note('Ce que le résultat établit', 'Les candidates cataloguées se concentrent davantage dans le voisinage des autres candidates que dans celui de leurs références comparables. Cela montre une association avec la sélection publiée. On ne mesure pas encore la capacité à découvrir de nouvelles lentilles.')}
{details('Comment vérifier que le contraste ne vient pas seulement des proportions ?',fig('neighbours_permutation','Contrôle par permutation : les étiquettes candidate et référence sont réattribuées dans chaque paire. Il s’agit d’une référence conditionnelle, pas d’un intervalle de confiance sur la découverte de lentilles.')+'''<p>Le graphe reste fixe ; à chaque permutation, on recalcule les labels des voisins et le contraste. Ce contrôle suppose que les rôles soient approximativement échangeables à l’intérieur des paires. L’appariement observationnel ne garantit pas cette hypothèse.</p>''')}
{details('Quel Zoobot est utilisé ?', '''<p>La source est la publication de morphologie Galaxy Zoo Euclid : 512 caractéristiques d’un réseau Zoobot gelé, réduites à 40 coordonnées par les auteurs. Ce réseau a été préentraîné avec des annotations de morphologie. Notre analyse de voisinage ne réentraîne pas ce réseau et n’utilise pas le détecteur affiné du papier C.</p><p>Les 40 nombres résument des propriétés apprises ; ce ne sont pas 40 mesures physiques nommées. Aucun autre checkpoint n’est utilisé pour compléter les représentations manquantes.</p>''')}
</section>
<section id="comparison"><div class="heading"><span>04</span><div><h2>Zoobot ou un modèle de vision généraliste ?</h2><p>Refaire la même comparaison avec un second traitement d’images.</p></div></div>
<p class="small">J’ai calculé les représentations d’un <strong>DINOv2 ViT-S/14 gelé</strong>, sans entraînement local. Sur 60 paires choisies à l’avance, 59 ont deux images exploitables : <strong>les 118 mêmes objets</strong> sont comparés avec les deux traitements.</p>
{fig('second_comparison','Même sous-échantillon et même calcul des voisins. Le contraste mesure l’écart entre candidates et références dans chaque espace.')}
{note('Zoobot est plus adapté à cet objectif dans notre protocole', f"Sur ces mêmes objets, le contraste est de <b>{pts(z['difference'])} points avec Zoobot</b>, contre <b>{pts(dn['difference'])} avec DINOv2</b>. Les représentations Zoobot rapprochent donc davantage les candidates cataloguées entre elles. Ce constat est cohérent avec l’intérêt d’un préentraînement sur la morphologie des galaxies.")}
<p class="small">Zoobot utilise les entrées VIS + Y des auteurs ; DINOv2 nos vignettes VIS seules, avec un autre cadrage et une autre normalisation. Le catalogue est aussi issu d’une présélection par plusieurs modèles, dont un détecteur dérivé de Zoobot. <strong>On ne démontre pas une supériorité générale des modèles spécialisés.</strong></p>
{details('Le biais de sélection : Zoobot a-t-il déjà appris sur ces images ?', '''<p>Le biais signalé concerne la construction du catalogue. Un détecteur adapté à la recherche de lentilles à partir de Zoobot a participé, avec d’autres modèles, à choisir les images proposées aux volontaires puis aux experts. Certaines formes bien repérées par cette famille de modèles peuvent donc être surreprésentées parmi les candidates.</p><p>Les représentations que nous utilisons viennent d’un autre modèle : un extracteur de morphologie gelé, préentraîné hors Euclid selon la publication. Le problème évoqué n’est donc pas un entraînement sur nos images Q1. Retrouver les candidates proches entre elles décrit utilement ce catalogue, mais ne prouve pas une capacité à détecter toutes les lentilles d’un échantillon indépendant.</p>''')}
{details('Le protocole DINOv2 en quelques lignes',f'''<p>20 paires par grade ont été choisies avec une graine fixe avant l’inspection des images. Une référence VIS inutilisable entraîne l’exclusion de sa paire entière, sans remplacement : 20 A, 20 B et 19 C restent. Ce sous-échantillon a une composition différente du corpus principal ; on compare ses deux traitements entre eux.</p><p>DINOv2 produit 384 caractéristiques, normalisées en longueur. Ses entrées sont des images VIS 96×96, étirées en contraste, répliquées en trois canaux, redimensionnées puis recadrées ; le champ effectif couvre 8,4″. L’inférence initiale a pris {e['execution']['inference_seconds']:.1f} secondes sur CPU. Les vecteurs, poids du modèle et sommes de contrôle sont conservés. Aucun entraînement supervisé local n’a été effectué.</p>''')}
</section>
<section id="conclusion"><div class="heading"><span>05</span><div><h2>Ce que je retiens de l’expérience.</h2><p>Une comparaison utile, avec un périmètre clair.</p></div></div>
<blockquote>« J’ai comparé chaque candidate publiée à un objet de brillance et de taille proches, dans la même région du ciel. Dans l’espace Zoobot, les voisins des candidates contiennent davantage d’autres candidates connues. Ce contraste est plus marqué avec Zoobot qu’avec DINOv2 sur les mêmes objets. C’est un argument pour étudier les représentations astronomiques, avant de prétendre détecter de nouvelles lentilles. »</blockquote>
<div class="cards"><article><h3>Le travail réalisé</h3><p>Acquisition de données publiques, contrôle des identifiants et des mesures, lecture d’images FITS, construction de paires, recherche de voisins, inférence d’un Transformer préentraîné et comparaison reproductible.</p></article><article><h3>La prochaine validation</h3><p>Inspecter indépendamment les objets proposés, obtenir des labels fiables et tester sur une autre région du ciel. Pour isoler l’effet du réseau : comparer les modèles avec les mêmes bandes et le même prétraitement.</p></article></div>
<ul class="limits"><li>Les références peuvent contenir de vraies lentilles : elles ne sont pas des négatifs confirmés.</li><li>Les candidates ont déjà été présélectionnées, notamment par des méthodes liées à Zoobot.</li><li>Les résultats concernent le corpus retenu, avec ses proportions construites et ses dépendances locales.</li></ul>
{details('Autres pistes explorées', '''<p>K-means, HDBSCAN et une recherche d’anomalies LOF ont aussi été examinés. HDBSCAN n’a retenu aucun groupe avec les paramètres fixés ; les anomalies montraient notamment des traînées et des aigrettes. Ces essais ont été écartés du parcours final pour le centrer sur la comparaison des voisins.</p>''')}
<p class="small">Deux notebooks exécutés : <b>01</b> données et paires comparables · <b>02</b> voisins et comparaison des représentations. Le README en anglais donne les commandes de reproduction.</p>
</section></main>
<dialog id="image-viewer" aria-label="Figure agrandie"><button type="button" id="close-image" aria-label="Fermer la figure">Fermer ×</button><img id="large-image" alt=""><p id="large-caption"></p></dialog>
<script>
const viewer=document.getElementById('image-viewer'),large=document.getElementById('large-image'),caption=document.getElementById('large-caption');
document.querySelectorAll('figure img').forEach(im=>{{im.tabIndex=0;im.setAttribute('role','button');im.setAttribute('aria-label','Agrandir : '+im.alt);function open(){{large.src=im.src;large.alt=im.alt;caption.textContent=im.closest('figure').querySelector('figcaption').textContent;viewer.showModal()}}im.addEventListener('click',open);im.addEventListener('keydown',ev=>{{if(ev.key==='Enter'||ev.key===' '){{ev.preventDefault();open()}}}})}});
document.getElementById('close-image').addEventListener('click',()=>viewer.close());
viewer.addEventListener('click',ev=>{{if(ev.target===viewer)viewer.close()}});
</script>
<footer><b>LensLab · Mohamed Bourega · Projet indépendant</b><p><a href="https://eas.esac.esa.int/">ESA Euclid Q1</a> · <a href="https://zenodo.org/records/15025832">Discovery Engine v0.0.3</a> · <a href="https://zenodo.org/records/15106473">Galaxy Zoo Euclid</a> · <a href="https://arxiv.org/abs/2503.15310">Publication de morphologie</a> · <a href="https://arxiv.org/html/2503.15326v2">Papier C</a> · <a href="https://huggingface.co/facebook/dinov2-small">DINOv2</a></p><p>Crédit aux équipes ESA, Euclid, Galaxy Zoo, aux volontaires et aux auteurs des catalogues et modèles.</p></footer></body></html>'''
    destination=root/'portfolio.html'
    body = body.replace('</body>', '<script>'+GALLERY_JS+'</script></body>')
    destination.write_text(CSS.replace('</style>', GALLERY_CSS+'</style>')+body,encoding='utf-8')
    return destination


GALLERY_JS = r'''
const galleryData = JSON.parse(document.getElementById('gallery-data').textContent);
const pairSelect = document.getElementById('gallery-pair');
const labelsToggle = document.getElementById('gallery-labels');
const galleryViews = document.getElementById('gallery-views');
const escapeGallery = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function galleryCard(id, rank, labels, anchor) {
  const item = galleryData.items[id];
  const labelled = item.label !== 'Unlabelled';
  const grade = labelled ? item.label : 'U';
  const status = labelled ? 'Candidate '+item.label : 'Sans label';
  const label = labels ? status : 'Label masqué';
  const title = 'Objet '+id+' · '+label;
  const picture = item.image_available && item.image
    ? `<button class="stamp" type="button" data-object="${escapeGallery(id)}" aria-label="Agrandir ${escapeGallery(title)}"><img src="${item.image}" alt="Image VIS · ${escapeGallery(title)}" loading="lazy"><span class="stamp-scale" aria-label="2 secondes d’arc">2″</span></button>`
    : '<div class="stamp unavailable">Image indisponible</div>';
  return `<article class="gallery-card ${anchor?'anchor-card':''}" title="${escapeGallery(title)}">${picture}<div class="stamp-caption"><span class="stamp-rank">${anchor?'Départ':'#'+rank}</span><span class="grade ${labels?'grade-'+grade:'grade-U'}">${escapeGallery(label)}</span></div></article>`;
}
function renderGalleryPair(pair, labels) {
  return ['candidate','reference'].map(role => {
    const candidate = role === 'candidate';
    const name = labels ? (candidate ? 'Départ : candidate '+pair.grade : 'Départ : référence sans label') : (candidate ? 'Départ : objet 1' : 'Départ : objet 2');
    const count = pair[role+'_labelled_count'];
    const score = labels ? `<strong>${count}/10</strong> <span>voisines candidates</span>` : '<span>Comptage masqué</span>';
    const neighbours = pair[role+'_neighbours'].map(n => galleryCard(n.object_id,n.rank,labels,false)).join('');
    return `<div class="gallery-row"><div class="gallery-row-heading"><h4>${name}</h4><div class="gallery-count">${score}</div></div><div class="gallery-image-row"><div class="gallery-anchor">${galleryCard(pair[role+'_id'],null,labels,true)}</div><div class="gallery-arrow" aria-hidden="true">→</div><div class="gallery-neighbours">${neighbours}</div></div></div>`;
  }).join('');
}
function updateGallery() {
  Array.from(pairSelect.options).forEach((option, i) => {
    option.textContent = 'Paire '+(i+1)+(labelsToggle.checked ? ' · candidate de grade '+galleryData.pairs[i].grade : '');
  });
  galleryViews.innerHTML = renderGalleryPair(galleryData.pairs[Number(pairSelect.value)], labelsToggle.checked);
  document.getElementById('gallery-legend').hidden = !labelsToggle.checked;
}
pairSelect.addEventListener('change',updateGallery);
labelsToggle.addEventListener('change',updateGallery);
galleryViews.addEventListener('click', event => {
  const button = event.target.closest('button[data-object]');
  if (!button) return;
  const id = button.dataset.object, item = galleryData.items[id];
  const status = labelsToggle.checked ? (item.label === 'Unlabelled' ? 'Sans label de candidate' : 'Candidate '+item.label) : 'Label masqué';
  const viewer = document.getElementById('image-viewer');
  const large = document.getElementById('large-image');
  large.src = item.image;
  large.alt = 'Euclid VIS · '+id+' · '+status;
  document.getElementById('large-caption').textContent = 'Objet '+id+' · '+status+' · VIS, champ de 9,6″. Agrandissement de la même vignette de 96 × 96 pixels.';
  viewer.showModal();
});
updateGallery();
'''

GALLERY_CSS = '''
.gallery-jump{display:inline-block;background:var(--teal);color:white;padding:11px 17px;border-radius:5px;font-size:14px;font-weight:650;text-decoration:none}
.gallery{background:white;border:1px solid var(--line);border-top:4px solid var(--teal);border-radius:6px;padding:25px;margin:30px 0;scroll-margin-top:20px}.gallery-heading .eyebrow{margin:0 0 7px}.gallery-heading h3{font-size:25px;line-height:1.25;margin:0 0 10px;letter-spacing:-.5px}.gallery-heading p:not(.eyebrow){font-size:14px;color:#526b7b;max-width:850px;margin:0 0 20px}.gallery-controls{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:16px;padding-block:16px;border-block:1px solid var(--line);font-size:13px}.gallery-controls select{font:inherit;color:var(--ink);background:#f8faf9;border:1px solid #abbcc5;border-radius:4px;padding:8px 10px;margin-left:10px}.label-toggle{display:flex;align-items:center;gap:7px;cursor:pointer}.label-toggle input{accent-color:var(--teal);width:17px;height:17px}.gallery-legend{display:flex;flex-wrap:wrap;gap:8px;margin:17px 0}.gallery-legend[hidden]{display:none}.gallery-legend>span,.grade{border-radius:3px;font-size:11px;padding:3px 6px;font-weight:650;white-space:nowrap}.grade-A{color:#066a61;background:#e0f3eb}.grade-B{color:#255686;background:#e7eef7}.grade-C{color:#835309;background:#faf0d7}.grade-U{color:#536671;background:#edf1f2}.gallery-row{padding:17px 0 22px;border-bottom:1px solid var(--line)}.gallery-row-heading{display:flex;justify-content:space-between;gap:12px;align-items:baseline;margin-bottom:14px}.gallery-row-heading h4{font-size:16px;margin:0}.gallery-count strong{font-size:23px;color:var(--teal);line-height:1}.gallery-count span{font-size:12px;color:var(--muted)}.gallery-image-row{display:grid;grid-template-columns:140px 28px 1fr;gap:14px;align-items:center}.gallery-arrow{font-size:23px;color:#6c8792;text-align:center}.gallery-neighbours{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px}.gallery-card{min-width:0}.stamp{display:block;position:relative;width:100%;aspect-ratio:1;border:0;padding:0;background:#07131e;overflow:hidden;border-radius:3px;cursor:zoom-in}.stamp img{display:block;width:100%;height:100%;object-fit:contain;image-rendering:auto}.stamp:focus-visible{outline:3px solid #1b9fb0;outline-offset:3px}.stamp-scale{position:absolute;bottom:9px;left:10px;width:20.833333%;border-bottom:2px solid white;color:white;text-align:center;font-size:10px;line-height:1.6;text-shadow:0 1px 3px black}.stamp-caption{display:flex;gap:5px;align-items:center;margin-top:6px;justify-content:space-between;flex-wrap:wrap}.stamp-rank{color:var(--muted);font-size:11px}.anchor-card{padding:8px;background:#edf5f1;border:1px solid #bed8ce;border-radius:5px}.unavailable{display:flex;align-items:center;justify-content:center;text-align:center;color:#cad6dd;font-size:12px;padding:10px;cursor:default}.gallery-note{font-size:12px;color:#5d7280;margin:16px 0 0;max-width:960px}
@media(max-width:900px){.gallery{padding:19px}.gallery-image-row{grid-template-columns:115px 18px 1fr;gap:10px}.gallery-neighbours{gap:9px}.grade{font-size:10px;padding:2px 4px}}
@media(max-width:650px){.gallery{padding:15px}.gallery-controls select{display:block;margin:7px 0 0}.gallery-row-heading{display:block}.gallery-count{margin-top:6px}.gallery-image-row{grid-template-columns:1fr}.gallery-anchor{width:142px;margin:0 auto}.gallery-arrow{transform:rotate(90deg);line-height:1;margin:4px}.gallery-neighbours{grid-template-columns:repeat(5,minmax(0,1fr));gap:6px}.stamp-caption{display:block;line-height:1.4}.stamp-rank{display:block;margin-bottom:3px}.grade{font-size:9px;white-space:normal;display:block;padding:3px}.gallery-heading h3{font-size:22px}.stamp-scale{font-size:8px;left:5px;bottom:5px}}
@media(max-width:430px){.gallery-neighbours{grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.stamp-caption{display:flex}.grade{font-size:11px}.stamp-rank{margin-bottom:0}}
@media print{.gallery-controls,.gallery-jump{display:none}.gallery-row{break-inside:avoid}}
'''


CSS='''<!doctype html><html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LensLab — Mohamed Bourega</title><meta name="description" content="Euclid : candidates, références comparables, voisins Zoobot et comparaison DINOv2. Portfolio de Mohamed Bourega."><style>
:root{--ink:#122b45;--muted:#62768a;--teal:#087e79;--line:#dce5e7;--paper:#f8faf9}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}a{color:var(--teal);text-underline-offset:4px}header{max-width:1260px;margin:auto;display:flex;justify-content:space-between;align-items:center;padding:25px 38px;border-bottom:1px solid var(--line);gap:20px}.brand{font-weight:800;font-size:22px;letter-spacing:-.6px;text-decoration:none;color:var(--ink)}.brand span{font-size:15px;font-weight:400;color:var(--muted)}nav{display:flex;gap:25px;font-size:13px}nav a{text-decoration:none;color:var(--muted)}main{max-width:1196px;margin:auto;padding:0 38px}.hero{display:grid;grid-template-columns:1.08fr 1fr;align-items:center;gap:45px;padding:60px 0 30px}.eyebrow{color:var(--teal);font-size:11px;font-weight:750;letter-spacing:1.5px;margin-bottom:24px}h1{font-size:clamp(38px,4.0vw,55px);line-height:1.07;letter-spacing:-2.7px;margin:0 0 25px}h1 em{font-style:normal;color:var(--teal)}.lead{font-size:18px;color:#456075;max-width:465px}.tags{display:flex;gap:8px;flex-wrap:wrap;margin-top:25px}.tags span{border:1px solid var(--line);border-radius:4px;padding:5px 9px;font-size:10px;color:#4a626e}.space{background:#091a2a;border-radius:5px;padding:18px;box-shadow:0 20px 50px #122b4520}.space img{width:100%;height:390px;object-fit:contain;display:block}.space small{display:block;color:#b5cbd5;letter-spacing:1px;font-size:10px;margin:14px 9px 5px}.intro{display:grid;grid-template-columns:1fr 1fr;gap:50px;color:#476074;margin:15px 0 26px}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:20px;border-block:1px solid var(--line);padding:24px 0;margin:30px 0}.stats strong{display:block;font-size:34px;letter-spacing:-1px;line-height:1.4}.stats span{font-size:12px;color:var(--muted)}.take{background:#edf5f1;border-left:3px solid var(--teal);padding:20px 25px;margin:26px 0}.take b{font-size:16px}.take p{margin:7px 0 0;font-size:14px;color:#3f6061}section[id]{padding:40px 0 8px;scroll-margin-top:20px}.heading{display:flex;gap:22px;margin:18px 0 28px}.heading>span{font-size:13px;color:var(--teal);font-weight:750;padding-top:8px}.heading h2{font-size:30px;line-height:1.25;letter-spacing:-.8px;margin:0 0 7px}.heading p{font-size:15px;color:var(--muted);margin:0}.cards{display:grid;grid-template-columns:1fr 1fr;gap:22px;margin:25px 0}.cards article{background:white;border:1px solid var(--line);border-radius:5px;padding:23px 26px}.cards h3{font-size:17px;margin:0 0 9px}.cards p{font-size:14px;color:#4e6777;margin:0}figure{margin:28px 0;padding:10px 10px 6px;background:white;border:1px solid #e3eaeb;border-radius:5px}figure img{display:block;width:100%;height:auto}figcaption{padding:8px 22px 15px;font-size:12px;color:var(--muted);max-width:990px}details{border-block:1px solid var(--line);padding:16px 3px;margin:25px 0}summary{cursor:pointer;font-size:14px;font-weight:650}details p{font-size:14px;max-width:950px;color:#526b7b}.small{font-size:14px;max-width:950px;color:#516a7b}.pipeline{display:grid;grid-template-columns:repeat(4,1fr);border:1px solid var(--line);background:white;border-radius:5px;margin:30px 0}.pipeline>div{padding:22px 24px;position:relative;border-right:1px solid var(--line)}.pipeline>div:last-child{border:0}.pipeline>div:not(:last-child):after{content:'→';position:absolute;right:-9px;top:35%;background:white;color:var(--teal);font-size:19px;z-index:1}.pipeline b{display:block;color:var(--teal);font-size:14px}.pipeline span{display:block;font-size:12px;color:var(--muted);margin-top:8px}table{width:100%;border-collapse:collapse;font-size:13px;margin:27px 0}th,td{text-align:left;padding:13px 12px;border-bottom:1px solid var(--line);vertical-align:top}th{color:var(--teal);font-size:12px}td:first-child{width:25%}.result{background:var(--ink);color:white;border-radius:5px;padding:30px 34px;margin:28px 0}.result .eyebrow{color:#abe0cb}.result .stats{grid-template-columns:repeat(3,1fr);border-color:#ffffff33;margin:15px 0;padding:21px 0}.result .stats strong{color:#bce2d2}.result .stats span{color:#c7d7e0}.result>p:last-child{font-size:14px;color:#d4e3e7}blockquote{font-size:18px;line-height:1.8;border-left:3px solid var(--teal);padding:12px 26px;margin:28px 0;background:white}code{font-size:.85em;background:#edf1f3;padding:2px 4px;overflow-wrap:anywhere}footer{background:#edf2f1;border-top:1px solid var(--line);margin-top:45px;padding:30px max(38px,calc((100vw - 1120px)/2));font-size:12px;color:var(--muted)}footer b{color:var(--ink);font-size:14px}
figure img{cursor:zoom-in}figure img:focus-visible{outline:3px solid var(--teal);outline-offset:3px}dialog{max-width:96vw;width:1500px;max-height:94vh;border:0;border-radius:7px;padding:45px 15px 12px;color:var(--ink);box-shadow:0 20px 70px #0005}dialog::backdrop{background:#091a2adc}dialog img{width:100%;height:auto;display:block}dialog p{font-size:13px;color:var(--muted);margin:12px 15px}dialog button{position:absolute;right:16px;top:12px;background:#edf5f1;border:1px solid var(--line);color:var(--teal);border-radius:4px;padding:6px 12px;cursor:pointer}
@media(max-width:760px){header{padding:20px;flex-wrap:wrap}nav{gap:20px}main{padding:0 20px}.hero{grid-template-columns:1fr;gap:25px;padding-top:35px}.space img{height:320px}.intro,.cards{grid-template-columns:1fr;gap:16px}.intro p{margin:6px 0}.stats{grid-template-columns:repeat(2,1fr)}h1{letter-spacing:-1.7px}.heading h2{font-size:25px}.heading{gap:12px}.pipeline{grid-template-columns:repeat(2,1fr)}.pipeline>div{padding:18px}.pipeline>div:nth-child(2):after{display:none}.result{padding:23px}.result .stats{grid-template-columns:1fr;gap:13px}.result .stats strong{font-size:30px}figure{padding:4px}figcaption{padding:10px}td,th{padding:10px 6px;font-size:12px}blockquote{font-size:16px;padding:15px 20px}.take{padding:18px 20px}footer{padding:25px 20px}}
@media print{nav{display:none}body{background:white}figure,.cards,.stats,.take{break-inside:avoid}.result{background:#edf5f1;color:var(--ink)}.result *{color:var(--ink)!important}.hero{padding-top:25px}details{display:none}footer{margin-top:20px}}
.overview{grid-template-columns:repeat(3,1fr)}.criteria{display:grid;grid-template-columns:repeat(3,1fr);border:1px solid var(--line);background:white;border-radius:5px;margin:25px 0}.criteria>div{padding:20px 23px;border-right:1px solid var(--line)}.criteria>div:last-child{border:0}.criteria b{display:block;font-size:15px;color:var(--teal)}.criteria span{display:block;font-size:13px;color:var(--muted);margin-top:7px}.experiment{margin:30px 0;border:1px solid var(--line);border-radius:6px;overflow:hidden;background:white}.experiment-start,.experiment-end{text-align:center;padding:21px;background:#edf5f1}.experiment-start b,.experiment-end b{display:block;font-size:16px}.experiment-start span,.experiment-end span{font-size:13px;color:#4b6b6b}.experiment-branches{display:grid;grid-template-columns:1fr 1fr}.experiment-branches>div{padding:28px 22px;text-align:center;border-right:1px solid var(--line)}.experiment-branches>div:last-child{border:0}.experiment-branches b,.experiment-branches span{display:block}.experiment-branches b{font-size:16px}.experiment-branches span{font-size:13px;color:var(--muted)}.experiment-branches strong{display:block;font-size:33px;color:var(--teal);line-height:1.3;margin:15px 0 4px}.experiment-branches>div:last-child strong{color:#708296}.limits{padding-left:21px;font-size:14px;color:#4e6777;line-height:1.8}@media(max-width:760px){.overview{grid-template-columns:repeat(3,1fr);gap:12px}.overview strong{font-size:27px}.criteria{grid-template-columns:1fr}.criteria>div{border-right:0;border-bottom:1px solid var(--line);padding:15px 20px}.criteria span{margin-top:2px}.experiment-branches>div{padding:22px 12px}.experiment-branches strong{font-size:28px}.experiment-branches b{font-size:14px}.experiment-start,.experiment-end{padding:18px 16px}}
</style></head>'''
