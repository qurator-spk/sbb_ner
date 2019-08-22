
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

                    if (url_params.has('ppn')) {

                        var ppn = url_params.get('ppn')

                        $('#ppn').val(ppn);

                        do_update = true;
                    }

                    if (url_params.has('model_id')) {

                        var model_id = url_params.get('model_id')

                        $('#model').val(model_id);

                        do_update = true;
                    }

                    if (url_params.has('task')) {

                        var task = url_params.get('task')

                        $('#task').val(task);

                        do_update = true;
                    }

                    task_select()

                    if (do_update) update();
                }
            );

    $.get( "/ppnexamples")
        .done(
            function( data ) {
                var tmp="";
                $.each(data,
                    function(index, item){

                        tmp += '<option value="' + item.ppn + '">' + item.name + '</option>'
                    });
                    $('#ppnexamples').html(tmp);
                }
            );
});


function update() {

    var spinner_html =
        `<div class="d-flex justify-content-center">
            <div class="spinner-border align-center" role="status">
                <span class="sr-only">Loading...</span>
            </div>
         </div>`;

    var task = $('#task').val();
    var model_id = $('#model').val();
    var ppn = $('#ppn').val();

    var url_params = new URLSearchParams(window.location.search);

    url_params.set('ppn', ppn)
    url_params.set('model_id', model_id)
    url_params.set('task', task)

    window.history.replaceState({}, '', `${location.pathname}?${url_params}`);

    $("#resultregion").html(spinner_html);

    $.get( "/digisam-fulltext/" + ppn)
        .done(function( data ) {

            do_task(task, model_id, data.text)
        })
        .fail(
            function() {
                console.log('Failed.');
                $("#resultregion").html('Failed.');
            });
}