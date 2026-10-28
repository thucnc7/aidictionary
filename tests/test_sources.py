from aidict.sources import parse_google, parse_wikipedia, parse_cheatsheet, merge_terms

GOOGLE_HTML = '''<p><a class="glossary-anchor" name="attention"></a>
<h2 class="hide-from-toc" id="attention" data-text=" attention" tabindex="-1"> attention</h2></p>
<p>A mechanism used in a <a href="#neural_network"><strong>neural network</strong></a> that indicates
the importance of a particular word or part of a word.</p>
<p>See <a href="/x">LLMs</a> in Machine Learning Crash Course for more information.</p>
<p><a class="glossary-anchor" name="attribute"></a>
<h2 class="hide-from-toc" id="attribute" data-text=" attribute" tabindex="-1"> attribute</h2>
<div class="glossary-icon-container">
  <div class="glossary-icon" title="Responsible AI">#responsible</div>
</div></p>
<p>Synonym for <a href="#feature"><strong>feature</strong></a>.</p>'''

WIKI_HTML = '''<dl><dt id="a*_search"><dfn><a href="/wiki/A*">A* search</a></dfn></dt>
<dd><p class="glossary-hatnote"><style>.x{}</style><span>Pronounced "A-star".</span></p>A graph traversal<sup>[1]</sup> algorithm.</dd></dl>'''

def test_parse_google():
    t = parse_google(GOOGLE_HTML)
    assert [x["term"] for x in t] == ["attention", "attribute"]
    assert t[0]["definition_en"] == "A mechanism used in a neural network that indicates the importance of a particular word or part of a word."
    assert t[1] == {"term": "attribute", "definition_en": "Synonym for feature.", "tags": ["responsible"], "source": "google"}

def test_parse_wikipedia():
    assert parse_wikipedia(WIKI_HTML) == [{"term": "A* search", "definition_en": 'Pronounced "A-star". A graph traversal algorithm.', "tags": [], "source": "wikipedia"}]

def test_parse_cheatsheet():
    md = "**1. Supervised Learning cheatsheet**\n\n&#10230; Cheatsheet học có giám sát\n\n<br>\n\n**2. Introduction to Supervised Learning**\n\n&#10230; Giới thiệu về học có giám sát\n"
    assert parse_cheatsheet(md) == [("Supervised Learning cheatsheet", "Cheatsheet học có giám sát"),
                                    ("Introduction to Supervised Learning", "Giới thiệu về học có giám sát")]

def test_merge_terms_dedupes_case_and_parenthetical():
    g = [{"term": "Transformer", "definition_en": "A", "tags": ["generativeAI"], "source": "google"}]
    w = [{"term": "transformer (deep learning)", "definition_en": "B", "tags": [], "source": "wikipedia"}]
    m = [{"term": "RAG", "definition_en": "", "tags": [], "source": "manual"}]
    out = merge_terms([g, w, m])
    assert out[0] == {"term": "Transformer", "definitions_en": ["A", "B"], "tags": ["generativeAI"], "sources": ["google", "wikipedia"]}
    assert out[1] == {"term": "RAG", "definitions_en": [], "tags": [], "sources": ["manual"]}
