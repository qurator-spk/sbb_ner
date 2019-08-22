$(document).ready(function(){

    $('#nerform').submit(
        function(e){
            e.preventDefault();

            update();
        }
    );

    $.get( "/models")
        .done(
            function( data ) {
                var tmp="";
                $.each(data,
                    function(index, item){

                        selected=""
                        if (item.default) {
                            selected = "selected"
                        }

                        tmp += '<option value="' + item.id + '" ' + selected + ' >' + item.name + '</option>'
                    });
                    $('#model').html(tmp);

                    var url_params = new URLSearchParams(window.location.search);

                    var do_update=false;

                    if (url_params.has('text')) {

                        var text = decodeURIComponent(url_params.get('text'))

                        $('#inputtext').val(text);

                        do_update = true;

                        window.history.replaceState({}, '', `${location.pathname}`);
                    }

                    task_select()

                    if (do_update) update();
                }
            );
});

function update() {

    var task = $('#task').val();
    var model_id = $('#model').val();
    var input_text = $('#inputtext').val()

    if (input_text.length < 200) {

        var url_params = new URLSearchParams(window.location.search);

        url_params.set('text', encodeURIComponent(input_text))

        window.history.replaceState({}, '', `${location.pathname}?${url_params}`);
    }



    do_task(task, model_id, input_text);
}