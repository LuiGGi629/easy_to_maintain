from src.transformers import CategoriesExtractor


def test_extract_categories():
    """
    json_string is getting passed by transformer, expecting attribute slug
    and then categories with schema main / sub categories. Then calling
    method and expecting the result to be list containing games and
    playing cards.
    """
    json_string = '{"slug": "games/playing cards"}'

    result = CategoriesExtractor.extract_categories(json_string, False)

    assert ['games', 'playing cards'] == result
