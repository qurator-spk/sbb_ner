
$(document).ready(function(){

    $('#nerform').submit(
        function(e){
            e.preventDefault();
            do_task();
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
                }
            );

    task_select()
});

function task_select() {

    var task = $('#task').val();

    if (task < 3) {
        $('#model_select').hide()
    }
    else {
        $('#model_select').show()
    }

    $("#resultregion").html("");
    $("#legende").html("");
}


function do_task() {

    var input_text = $('#inputtext').val()

    var text_region_html =
        `<div class="card">
            <div class="card-header">
                Ergebnis:
            </div>
            <div class="card-block">
                <div id="textregion" style="overflow-y:scroll;height: 65vh;"></div>
            </div>
        </div>`;

    var legende_html =
         `<div class="card">
            <div class="card-header">
                Legende:
                <div class="ml-2" >[<font color="red">Person</font>]</div>
                <div class="ml-2" >[<font color="green">Ort</font>]</div>
                <div class="ml-2" >[<font color="blue">Organisation</font>]</div>
                <div class="ml-2" >[keine Named Entity]</div>
            </div>
        </div>`;

    var spinner_html =
        `<div class="d-flex justify-content-center">
            <div class="spinner-border align-center" role="status">
                <span class="sr-only">Loading...</span>
            </div>
         </div>`;

    $("#legende").html("");

    var task = $('#task').val();
    var model_id = $('#model').val();

//    if (task == 2) {
//        $("#resultregion").html(spinner_html);
//
//        $.get( "/digisam-tokenized/" + ppn,
//            function( data ) {
//                $("#resultregion").html(text_region_html)
//                $("#textregion").html(data.text)
//            }).fail(
//            function() {
//                console.log('Failed.')
//                $("#resultregion").html('Failed.')
//            });
//    }
//    else
//
    if (task == 3) {

        $("#resultregion").html(spinner_html);

        post_data = { "text" : input_text }

        console.log(post_data)

        $.ajax({
            url:  "/ner/" + model_id,
            data: JSON.stringify(post_data),
            type: 'POST',
            contentType: "application/json",
            success:
                function( data ) {
                    text_html = ""
                    data.forEach(
                        function(sentence) {
                            sentence.forEach(
                                function(token) {

                                     if (text_html != "") text_html += ' '

                                     if (token.prediction == 'O')
                                        text_html += token.word
                                     else if (token.prediction.endsWith('PER'))
                                        text_html += '<font color="red">' + token.word + '</font>'
                                     else if (token.prediction.endsWith('LOC'))
                                        text_html += '<font color="green">' + token.word + '</font>'
                                     else if (token.prediction.endsWith('ORG'))
                                        text_html += '<font color="blue">' + token.word + '</font>'
                                })
                             text_html += '<br/>'
                        }
                    )
                    $("#resultregion").html(text_region_html)
                    $("#textregion").html(text_html)
                    $("#legende").html(legende_html)
                }
            ,
            error: function(error) {
                console.log(error);
            }
        });


//        $.post( "/ner/" + model_id, post_data).done(
//            function( data ) {
//
//                text_region_html = ""
//                data.forEach(
//                    function(sentence) {
//                        sentence.forEach(
//                            function(token) {
//                                text_region_html += token.word + "(" + token.prediction + ") "
//                            })
//                    }
//                )
//
//                $("#resultregion").html(text_region_html)
//                $("#textregion").html(data.text)
//                $("#legende").html(legende_html)
//            }).fail(
//            function(a,b,c) {
//                console.log('Failed.')
//                $("#resultregion").html('Failed.')
//            });
     }
//     else
//
//     if (task == 4) {
//        $("#resultregion").html(spinner_html);
//
//        $.get( "/digisam-ner-bert-tokens/" + model_id + "/" + ppn,
//            function( data ) {
//                $("#resultregion").html(text_region_html)
//                $("#textregion").html(data.text)
//            }).fail(
//            function(a,b,c) {
//                console.log('Failed.')
//                $("#resultregion").html('Failed.')
//            });
//     }
}