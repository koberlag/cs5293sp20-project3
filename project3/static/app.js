
(function(){
    let ingredientList = [];
    
    function addIngredient(ingredient){
        // If the item is not empty and does not already exist
        if(ingredient.trim() != "" && !isDuplicate(ingredient)){
            clearIngredientTextBox();
            // Add to ingredient list
            ingredientList.push(ingredient);
            // Add to ul
            renderIngredientsToAnalyze();
        }
    }

    function removeIngredient(ingredient){
        ingredientList = ingredientList.filter(i => i != ingredient)
    }

    function createIngredientsToAnalyzeListItemElems(){
        const sortedIngredients = ingredientList.sort()
        return sortedIngredients.map(function(ingredient){
            return '<li class="ingredient_item">' + ingredient + '</li>'
        }).join('');
    }

    function renderIngredientsToAnalyze(){
       const listItems = createIngredientsToAnalyzeListItemElems();
       getIngredientsToAnalyzeListElem().innerHTML = listItems;
       toggleIngredientSection();
    }

    function toggleIngredientSection(){
        const ingredientSection = document.getElementById("ingredient_section");
        const hasNoItemsClass = ingredientSection.classList.contains("no_items");
        const hasItems = ingredientList.length > 0;
        if(hasItems && hasNoItemsClass){
            ingredientSection.classList.remove('no_items');
        }
        else if(!hasItems && !hasNoItemsClass){
            ingredientSection.classList.add('no_items');
        }
    }

    function isDuplicate(ingredient){
        const ingredients = getListOfIngredients();
        return ingredients.includes(ingredient)
    }

    function clearIngredientTextBox(){
        getIngredientInputElem().value = "";
    }

    function getListOfIngredients(){
        return ingredientList;
    }

    // HTML Element getters
    function getIngredientInputElem(){
        return document.getElementById("tb_ingredient");
    }

    function getIngredientsToAnalyzeListElem(){
        return document.getElementById("ingredients_to_analyze");
    }
   
     // Event Listeners
    getIngredientsToAnalyzeListElem().addEventListener('click', event =>{
        if(event.target.classList.contains('ingredient_item')){
            const ingredient = event.target.textContent;
            removeIngredient(ingredient);
            renderIngredientsToAnalyze();
        }
    });
   
    getIngredientInputElem().onkeypress = function(e) {
        const key = e.charCode || e.keyCode || 0;     
        if (key == 13) {
            e.preventDefault();
            addIngredient(getIngredientInputElem().value);
        }
    }

    document.getElementById("ingredient_form").addEventListener('submit', onSubmitHandler)

    function onSubmitHandler(e){
        // Set hidden input to have all ingredient values to be submitted
        let hdnAllIngredients = document.getElementById("hdn_all_ingredients");
        hdnAllIngredients.value = ingredientList;
    }
})();
